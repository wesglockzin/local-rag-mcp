#!/usr/bin/env python3
"""
ingest.py — idempotent ingest of the corpus into the local Chroma store.

  - mtime-skip for unchanged files (with a tolerance compare: Chroma stores
    metadata floats at lower precision than getmtime returns, and an exact
    equality re-embeds a slice of the corpus on every run).
  - EMBED-THEN-SWAP: a file's old chunks are deleted only after every new
    chunk embedded successfully — an Ollama failure mid-file must never
    leave that file missing from the index.
  - Files deleted from disk (or newly `rag: false`) drop out via the sweep.
  - The corpus's git commit is stamped into the run output, so an index
    build can be pinned to exactly the corpus state that produced it.

Run: python ingest.py [--force]
Config via env: RAG_CORPUS_DIR, RAG_STORE_DIR, OLLAMA_HOST and RAG_EMBED_MODEL
(the last two are required; see common.py and .env.example).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

from common import (COLLECTION, CORPUS_DIR, MAX_CHARS_EMBED, STORE_DIR,
                    ConfigError, EmbedError, chunk_mode, corpus_files,
                    doc_prefix, embed, embed_model, heading_chunks,
                    ollama_host, parse_frontmatter, rag_enabled,
                    strip_frontmatter)

log = logging.getLogger("ingest")


def get_collection():
    import chromadb
    from chromadb.config import Settings
    os.makedirs(STORE_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=STORE_DIR, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})


def existing_mtimes(col) -> dict[str, float]:
    out: dict[str, float] = {}
    got = col.get(include=["metadatas"])
    for md in got["metadatas"] or []:
        p, m = md.get("path"), float(md.get("mtime", 0))
        if p:
            out[p] = max(out.get(p, 0), m)
    return out


def corpus_commit() -> str:
    """Reproducibility stamp — 'uncommitted changes present' is a warning
    label, and 'unknown' must never crash an ingest over a diagnostic."""
    try:
        dirty = subprocess.run(["git", "-C", CORPUS_DIR, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        sha = subprocess.run(["git", "-C", CORPUS_DIR, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if not sha:
            return "unknown (corpus is not a git repo)"
        return f"{sha} (uncommitted changes present)" if dirty else sha
    except (OSError, subprocess.SubprocessError) as e:
        return f"unknown ({type(e).__name__})"


def main() -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    force = "--force" in sys.argv
    try:
        ollama_host()
        embed_model()
    except ConfigError as e:
        log.error("%s", e)
        return 2
    col = get_collection()
    known = existing_mtimes(col)
    seen: set[str] = set()
    added = skipped = removed = failed = 0

    for path in corpus_files():
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        fm = parse_frontmatter(text)
        if not rag_enabled(fm):
            continue  # exclusion; index removal happens in the sweep
        seen.add(path)
        mtime = os.path.getmtime(path)
        if not force and abs(known.get(path, -1.0) - mtime) < 0.001:
            skipped += 1
            continue
        body = strip_frontmatter(text)  # embed content, not YAML syntax
        if chunk_mode(fm) == "whole":
            chunks = [{"text": body[:MAX_CHARS_EMBED], "heading_path": ""}]
        else:
            chunks = heading_chunks(body)
        ids, embs, docs, metas = [], [], [], []
        ok = True
        for i, ch in enumerate(chunks):
            # Doc-side regime prefix: whole files title by their filename stem;
            # heading chunks title by their heading path. Prefix AFTER
            # truncation so the instruction prefix is never cut.
            if chunk_mode(fm) == "whole":
                title = os.path.basename(path)[:-3].replace("_", " ")
            else:
                title = ch["heading_path"] or os.path.basename(path)
            try:
                vec = embed(doc_prefix(title) + ch["text"][:MAX_CHARS_EMBED])
            except EmbedError as e:
                log.error("  !! embed failed: %s chunk %d — keeping old index entries: %s",
                          path, i, e)
                ok = False
                failed += 1
                break
            ids.append(f"{path}::{i}")
            embs.append(vec)
            docs.append(ch["text"][:MAX_CHARS_EMBED])
            metas.append({"path": path, "chunk": i, "mtime": mtime,
                          "heading_path": ch["heading_path"],
                          "real_path": os.path.realpath(path),
                          "name": fm.get("name", os.path.basename(path)),
                          # Written explicitly for EVERY chunk: the search-side
                          # superseded pre-filter stands on this key, and a
                          # missing key is not a safe default under $ne on all
                          # Chroma versions. A re-ingest that dropped this
                          # silently re-exposed retired docs in the original —
                          # found by a regression test, kept by this line.
                          "lifecycle": fm.get("lifecycle", "current")})
        if ok and ids:
            # UPSERT-then-prune, not delete-then-add: a failure between a
            # delete and an add would lose the file from the index entirely
            # (adversarial-review finding). Upsert replaces chunks in place;
            # only chunks beyond the new count (a file that shrank) are
            # deleted, and only after the write succeeded.
            old_ids = set((col.get(where={"path": path}) or {}).get("ids") or [])
            col.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
            stale = sorted(old_ids - set(ids))
            if stale:
                col.delete(ids=stale)
            added += 1
            log.info("  + %s (%d chunk(s))", os.path.basename(path), len(ids))

    for path in set(known) - seen:
        col.delete(where={"path": path})
        removed += 1
        log.info("  - %s (removed)", os.path.basename(path))

    log.info("\ningest done: %d (re)indexed, %d unchanged, %d removed, %d failed",
             added, skipped, removed, failed)
    log.info("  store: %d chunks @ %s", col.count(), STORE_DIR)
    log.info("  corpus commit: %s", corpus_commit())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
