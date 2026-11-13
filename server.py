#!/usr/bin/env python3
"""
server.py — a read-only MCP server over a local RAG store.

Exposes two tools to any MCP client (Claude Code, Codex, anything speaking
the protocol), over stdio:

  search_corpus(query, k=4)  — semantic search: chunks + paths + cosine scores
  get_file(path)             — full text, restricted to the INDEXED corpus only

Everything runs on-device: Ollama embeds the query, Chroma holds the vectors,
nothing leaves the host. Built for environments where the corpus content
cannot go to a cloud API.

Contracts:
  - READ-ONLY. The server never creates or initializes stores — ingestion
    owns creation (ingest.py). A read-only server that quietly creates an
    empty store turns "you forgot to ingest" into "search returns nothing,"
    which is the worse failure (adversarial-review finding on the original).
  - Tool-level failures return structured {"error": ...} payloads, never
    exceptions — a down dependency must degrade the tool, not kill the session.
  - get_file is NOT a filesystem reader: only paths present in the index are
    readable, and a path that now resolves elsewhere than at ingest time
    (replaced / symlinked) is refused.

The tool logic lives in plain functions (do_search / do_get_file) so the test
suite exercises it directly, without a client or a model; the MCP layer is a
thin registration at the bottom.
"""

from __future__ import annotations

import logging
import os
import sys

from common import (
    COLLECTION,
    STORE_DIR,
    ConfigError,
    EmbedError,
    cosine_score,
    embed_model,
    embed_query,
    ollama_host,
)

log = logging.getLogger("server")

_collection = None


def get_collection():
    """Open the EXISTING collection only — never create (see module docstring)."""
    global _collection
    if _collection is None:
        # PersistentClient will happily initialize database files inside an
        # existing empty directory, which would make a "read-only" server a
        # writer. Require the database file itself, not just the directory.
        if not os.path.isfile(os.path.join(STORE_DIR, "chroma.sqlite3")):
            raise FileNotFoundError(f"store missing: {STORE_DIR} — run ingest.py first")
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=STORE_DIR,
            # "Nothing leaves the host" includes the vector store's own
            # telemetry, which defaults ON.
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(COLLECTION)
    return _collection


def do_search(query: str, k: int = 4) -> dict:
    try:
        q = str(query or "").strip()
        k = max(1, min(int(k), 20))
    except (TypeError, ValueError):
        return {"error": "invalid arguments: query=str, k=int required."}
    if len(q) < 3:
        return {"error": "query must be at least 3 characters."}
    try:
        vec = embed_query(q)
    except (ConfigError, EmbedError) as e:
        log.error("embedding failed: %s", e)
        return {"error": f"embedding unavailable: {e}. No search performed."}
    try:
        col = get_collection()
        if col.count() == 0:
            return {"query": q, "results": [], "note": "index is empty — run ingest.py"}
        # PRE-FILTER: retired docs are excluded by lifecycle metadata BEFORE
        # the knn search — a `where` clause narrows eligible ids first, so
        # this does not suffer the recall loss of post-filtering. Ingest
        # writes the lifecycle key explicitly on EVERY chunk; absence is not
        # a safe default under $ne on all Chroma versions.
        res = col.query(
            query_embeddings=[vec],
            n_results=min(k, col.count()),
            where={"lifecycle": {"$ne": "superseded"}},
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for doc, md, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append(
                {
                    "score": cosine_score(dist),
                    "path": md.get("path"),
                    "name": md.get("name"),
                    "heading_path": md.get("heading_path") or None,
                    "text": doc[:2000],
                }
            )
        return {"query": q, "results": results}
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001 - tool boundary: log, return structured error
        log.exception("search failed")
        return {"error": f"{type(e).__name__}: {e}"}


def do_get_file(path: str) -> dict:
    path = str(path or "")
    try:
        col = get_collection()
        # Membership check by exact path — not a full-index scan per call.
        got = col.get(where={"path": path}, include=["metadatas"])
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001 - tool boundary: log, return structured error
        log.exception("get_file lookup failed")
        return {"error": f"{type(e).__name__}: {e}"}
    mds = got["metadatas"] or []
    if not mds:
        return {
            "error": "path not in the indexed corpus.",
            "hint": "use search_corpus to discover valid paths.",
        }
    indexed = {path: mds[0].get("real_path") or path}
    if os.path.exists(path):
        # Symlink/replacement hardening: the file must still resolve to where
        # it resolved at ingest time, or an attacker who can swap a corpus
        # file for a symlink reads outside the corpus through this tool.
        if os.path.realpath(path) != indexed[path]:
            return {
                "error": "path now resolves elsewhere than at ingest "
                "(replaced/symlinked) — refusing to read. Re-run ingest."
            }
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            return {"path": path, "chars": len(text), "text": text[:50000]}
        except OSError as e:
            log.error("read failed for %s: %s", path, e)
            return {"error": f"read failed: {type(e).__name__}: {e}"}
    # File absent on disk (moved corpus, other machine): serve the indexed
    # chunk text rather than failing — the index has the content.
    try:
        got = col.get(where={"path": path}, include=["documents", "metadatas"])
        pairs = sorted(
            zip(got["metadatas"], got["documents"]), key=lambda p: p[0].get("chunk", 0)
        )
        chunks = [d for _, d in pairs]
    except Exception:  # noqa: BLE001 - chroma raises varied types; log, report "no text"
        log.exception("indexed-text lookup failed for %s", path)
        chunks = []
    if chunks:
        return {
            "path": path,
            "source": "index (file absent on disk)",
            "text": "\n\n".join(chunks)[:50000],
        }
    return {"error": "file absent and no indexed text found."}


# --- MCP registration (thin layer; logic above is what the tests exercise) ---


def build_server():
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations

    mcp = MCPServer(
        name="local-rag",
        instructions=(
            "Semantic search over a local document corpus (on-device "
            "embeddings; nothing leaves the host). Use search_corpus for "
            "'do we know anything about X' recall — results carry source "
            "paths and cosine scores; treat low scores as weak matches. "
            "get_file returns a full indexed document."
        ),
        version="1.0.0",
    )
    ro = ToolAnnotations(readOnlyHint=True, destructiveHint=False)

    @mcp.tool(annotations=ro)
    def search_corpus(query: str, k: int = 4) -> dict:
        """Semantic search of the corpus. Returns up to k chunks with source
        path, heading path (for chunked docs), cosine score, and text."""
        return do_search(query, k)

    @mcp.tool(annotations=ro)
    def get_file(path: str) -> dict:
        """Full text of an indexed document. Only paths present in the index
        are readable — this tool is not a general filesystem reader."""
        return do_get_file(path)

    return mcp


def main() -> int:
    # stdout carries the MCP protocol on stdio, so diagnostics go to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    try:
        ollama_host()
        embed_model()
    except ConfigError as e:
        log.error("%s", e)
        return 2
    build_server().run()  # stdio
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
