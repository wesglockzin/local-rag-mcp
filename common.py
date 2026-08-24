#!/usr/bin/env python3
"""
common.py — shared config + helpers for the local-rag-mcp substrate
(the ingest script and the MCP server import from here).

Design notes that are easy to miss:
  - The Chroma store is a live SQLite-backed database. It is machine-local by
    design: sync the CORPUS and this recipe, never the store — cloud sync does
    whole-file replacement with no transactional awareness, and the failure
    mode is a silently corrupted index on the machine that didn't write it.
    Every machine builds its own store from the same corpus.
  - Embedding regime: EmbeddingGemma uses asymmetric instruction prefixes —
    queries and documents embed differently, and Ollama passes text through
    raw, so the caller owns the prefixes. All query paths go through
    embed_query(); all document paths prefix with doc_prefix(). Mixing
    prefixed and raw vectors puts scores in a third, uncalibrated band.

Configuration (environment):
  RAG_CORPUS_DIR   directory of *.md files to index (default: ./sample-corpus)
  RAG_STORE_DIR    where the Chroma store lives (default: ~/.local-rag-mcp/store)
  OLLAMA_HOST      Ollama base URL (default: http://localhost:11434)
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
EMBED_MODEL = "embeddinggemma"

# Must match across ingest and search, and matches the companion project
# (claude-code-session-memory) exactly — same model, same regime, so scores
# and the calibrated floor are comparable across both consumers.
REGIME = "prefix-title-v1"
QUERY_PREFIX = "task: search result | query: "

CORPUS_DIR = os.path.abspath(os.environ.get("RAG_CORPUS_DIR", "sample-corpus"))
STORE_DIR = os.path.expanduser(os.environ.get("RAG_STORE_DIR", "~/.local-rag-mcp/store"))
COLLECTION = "corpus"

MAX_CHARS_EMBED = 6000


def doc_prefix(title: str) -> str:
    return f"title: {title} | text: "


def embed(text: str, timeout: float = 120.0) -> list[float] | None:
    """Embed via Ollama; newer /api/embed with /api/embeddings fallback."""
    for url, body, pick in (
        (f"{OLLAMA}/api/embed", {"model": EMBED_MODEL, "input": text},
         lambda d: (d.get("embeddings") or [None])[0] or d.get("embedding")),
        (f"{OLLAMA}/api/embeddings", {"model": EMBED_MODEL, "prompt": text},
         lambda d: d.get("embedding")),
    ):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                vec = pick(json.loads(r.read()))
            if vec:
                return vec
        except Exception:
            continue
    return None


def embed_query(text: str, timeout: float = 10.0) -> list[float] | None:
    """Query-side embed. ALL search paths use this — never raw embed()."""
    return embed(QUERY_PREFIX + text, timeout)


def parse_frontmatter(text: str) -> dict:
    """Minimal frontmatter reader — only the flat keys this project honors."""
    fm: dict = {}
    if not text.startswith("---"):
        return fm
    end = text.find("\n---", 3)
    if end < 0:
        return fm
    for line in text[3:end].splitlines():
        m = re.match(r"^\s*(rag|rag_chunk|name|description|lifecycle)\s*:\s*(.+?)\s*$", line)
        if m:
            val = m.group(1), m.group(2)
            # Strip YAML inline comments: `rag: false  # scratch notes` must
            # read as "false", not "false  # scratch notes" — otherwise every
            # frontmatter control silently stops matching its expected value
            # (adversarial-review finding; also patched in the companion repo).
            raw = re.split(r"\s+#", val[1], 1)[0]
            fm[val[0]] = raw.strip().strip('"').strip("'")
    return fm


def strip_frontmatter(text: str) -> str:
    """Body without the frontmatter block. Embedding the frontmatter pollutes
    search text with YAML syntax (a heading-split preamble chunk literally
    began with `---|name:` before this); the on-disk file keeps it — this
    strips only what gets embedded and served as chunk text."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end < 0:
        return text
    return text[end + 4:].lstrip("\n")


def rag_enabled(fm: dict) -> bool:
    """`rag: false` excludes a file from the index entirely."""
    return str(fm.get("rag", "true")).lower() not in ("false", "no", "0")


def chunk_mode(fm: dict) -> str:
    """`rag_chunk: headings` opts a long document into heading-split chunks;
    default is one whole-file embedding (right for small, atomic files)."""
    return "headings" if str(fm.get("rag_chunk", "whole")).lower() == "headings" else "whole"


def heading_chunks(text: str) -> list[dict]:
    """Split a document on markdown headings; carry the heading PATH as
    metadata. Chunks under 200 chars merge into the previous chunk; anything
    over the embed cap is sub-split with overlap so no content is silently
    invisible to search (a finding from adversarial review of the original)."""
    lines = text.splitlines(keepends=True)
    chunks: list[dict] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    cur_path = "(preamble)"

    def flush():
        body = "".join(buf).strip()
        if not body:
            return
        if chunks and len(body) < 200:
            chunks[-1]["text"] += "\n\n" + body
            return
        chunks.append({"text": body, "heading_path": cur_path})

    for ln in lines:
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            flush()
            buf = [ln]
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_path = " > ".join(t for _, t in stack)
        else:
            buf.append(ln)
    flush()

    out: list[dict] = []
    for ch in chunks:
        t = ch["text"]
        if len(t) <= MAX_CHARS_EMBED:
            out.append(ch)
            continue
        step = MAX_CHARS_EMBED - 200
        for j, start in enumerate(range(0, len(t), step)):
            out.append({"text": t[start:start + MAX_CHARS_EMBED],
                        "heading_path": f"{ch['heading_path']} [part {j+1}]"})
    return out


def corpus_files() -> list[str]:
    """Sorted absolute paths of every *.md in the corpus directory."""
    return [os.path.join(CORPUS_DIR, b)
            for b in sorted(os.listdir(CORPUS_DIR)) if b.endswith(".md")]


def cosine_score(distance: float) -> float:
    """Chroma cosine space returns distance = 1 - cosine_similarity."""
    return round(1.0 - distance, 4)
