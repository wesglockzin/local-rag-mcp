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
  OLLAMA_HOST      Ollama base URL (REQUIRED, no default)
  RAG_EMBED_MODEL  embedding model name (REQUIRED, no default)
"""

from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

# Must match across ingest and search. The prefixes below are EmbeddingGemma's
# documented regime; set RAG_EMBED_MODEL to a model that uses them.
REGIME = "prefix-title-v1"
QUERY_PREFIX = "task: search result | query: "

CORPUS_DIR = os.path.abspath(os.environ.get("RAG_CORPUS_DIR", "sample-corpus"))
STORE_DIR = os.path.expanduser(
    os.environ.get("RAG_STORE_DIR", "~/.local-rag-mcp/store")
)
COLLECTION = "corpus"

MAX_CHARS_EMBED = 6000


def doc_prefix(title: str) -> str:
    return f"title: {title} | text: "


class ConfigError(RuntimeError):
    """A required setting is missing."""


class EmbedError(RuntimeError):
    """The embedding server could not produce a vector; the message says why."""


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"required setting {name} is not set (see .env.example)")
    return value


def ollama_host() -> str:
    value = require_env("OLLAMA_HOST").rstrip("/")
    parts = urllib.parse.urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError(
            "setting OLLAMA_HOST must be a full URL starting with http:// or "
            "https:// (for example https://ollama.example.invalid:11434)"
        )
    if "@" in parts.netloc:
        raise ConfigError(
            "setting OLLAMA_HOST must not contain a user:password@ part; "
            "urllib does not send it as credentials"
        )
    try:
        _ = parts.port  # reading it validates the port
    except ValueError as e:
        raise ConfigError(f"setting OLLAMA_HOST has an invalid port: {e}") from e
    return value


def _redact(url: str) -> str:
    """Drop any user:password@ part so credentials never reach error text."""
    parts = urllib.parse.urlsplit(url)
    if "@" not in parts.netloc:
        return url
    netloc = parts.netloc.rsplit("@", 1)[1]
    return urllib.parse.urlunsplit(parts._replace(netloc=netloc))


def embed_model() -> str:
    return require_env("RAG_EMBED_MODEL")


def _post_json(url: str, body: dict, timeout: float) -> dict:
    shown = _redact(url)
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    except ValueError as e:
        raise EmbedError(f"invalid URL {shown}: {e}") from e
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise EmbedError(f"HTTP {e.code} from {shown}: {detail}") from e
    except (OSError, http.client.HTTPException) as e:
        # URLError, timeouts, refused connections, truncated responses
        raise EmbedError(f"cannot reach {shown}: {e}") from e
    try:
        return json.loads(raw)
    except ValueError as e:
        raise EmbedError(f"invalid JSON from {shown}: {e}") from e


def embed(text: str, timeout: float = 120.0) -> list[float]:
    """Embed via Ollama; newer /api/embed with /api/embeddings fallback.
    Raises EmbedError (with status and body for HTTP errors) on failure."""
    host, model = ollama_host(), embed_model()
    errors: list[str] = []
    for url, body, pick in (
        (
            f"{host}/api/embed",
            {"model": model, "input": text},
            lambda d: (d.get("embeddings") or [None])[0] or d.get("embedding"),
        ),
        (
            f"{host}/api/embeddings",
            {"model": model, "prompt": text},
            lambda d: d.get("embedding"),
        ),
    ):
        try:
            vec = pick(_post_json(url, body, timeout))
        except EmbedError as e:
            errors.append(str(e))
            continue
        if vec:
            return vec
        errors.append(f"no embedding in response from {url}")
    raise EmbedError("; ".join(errors))


def embed_query(text: str, timeout: float = 10.0) -> list[float]:
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
        m = re.match(
            r"^\s*(rag|rag_chunk|name|description|lifecycle)\s*:\s*(.+?)\s*$", line
        )
        if m:
            val = m.group(1), m.group(2)
            # Strip YAML inline comments: `rag: false  # scratch notes` must
            # read as "false", not "false  # scratch notes" — otherwise every
            # frontmatter control silently stops matching its expected value
            # (adversarial-review finding; also patched in the companion repo).
            raw = re.split(r"\s+#", val[1], maxsplit=1)[0]
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
    return text[end + 4 :].lstrip("\n")


def rag_enabled(fm: dict) -> bool:
    """`rag: false` excludes a file from the index entirely."""
    return str(fm.get("rag", "true")).lower() not in ("false", "no", "0")


def chunk_mode(fm: dict) -> str:
    """`rag_chunk: headings` opts a long document into heading-split chunks;
    default is one whole-file embedding (right for small, atomic files)."""
    return (
        "headings"
        if str(fm.get("rag_chunk", "whole")).lower() == "headings"
        else "whole"
    )


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
            out.append(
                {
                    "text": t[start : start + MAX_CHARS_EMBED],
                    "heading_path": f"{ch['heading_path']} [part {j + 1}]",
                }
            )
    return out


def corpus_files() -> list[str]:
    """Sorted absolute paths of every *.md in the corpus directory."""
    return [
        os.path.join(CORPUS_DIR, b)
        for b in sorted(os.listdir(CORPUS_DIR))
        if b.endswith(".md")
    ]


def cosine_score(distance: float) -> float:
    """Chroma cosine space returns distance = 1 - cosine_similarity."""
    return round(1.0 - distance, 4)
