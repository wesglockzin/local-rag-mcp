# local-rag-mcp

[![ci](https://github.com/wesglockzin/local-rag-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/wesglockzin/local-rag-mcp/actions/workflows/ci.yml)

> **A read-only MCP server for semantic search over a local document corpus —
> on-device embeddings (Ollama), a local Chroma store, nothing leaves the
> host.** Built for environments where the corpus content cannot go to a
> cloud API, and served identically to every MCP client (Claude Code, Codex,
> anything speaking the protocol).

This is the MCP-served sibling of
[claude-code-session-memory](https://github.com/wesglockzin/claude-code-session-memory):
same embedding model, same instruction-prefix regime, same measurement
methodology — one retrieval substrate, two consumers. The session-memory
README carries the full eval story (pre-committed bars, adversarial query
sets, regression attribution); this repo applies the same discipline to a
server instead of a hook.

## Tools

| Tool | What it does |
|---|---|
| `search_corpus(query, k=4)` | Semantic search: up to *k* chunks with source path, heading path, cosine score, text |
| `get_file(path)` | Text of an **indexed** document (capped at 50 k chars) — deliberately not a general filesystem reader |

Both are annotated read-only. Failures return structured `{"error": ...}`
payloads — a down dependency degrades the tool, never the session.

## Quickstart

```bash
git clone https://github.com/wesglockzin/local-rag-mcp
cd local-rag-mcp
python3 -m venv .venv && ./.venv/bin/pip install -e .
ollama pull embeddinggemma

# Required settings (no defaults): your Ollama URL and embedding model
export OLLAMA_HOST=<ollama base url>
export RAG_EMBED_MODEL=embeddinggemma

# Index the included sample corpus (or point RAG_CORPUS_DIR at your own)
./.venv/bin/local-rag-ingest

# Register with Claude Code — ABSOLUTE path, and pass the settings along: the
# MCP client launches the server from its own working directory and
# environment.
claude mcp add local-rag -e OLLAMA_HOST="$OLLAMA_HOST" -e RAG_EMBED_MODEL="$RAG_EMBED_MODEL" \
  -- "$PWD/.venv/bin/local-rag-server"
```

Then ask Claude Code something the corpus knows — "who gets paged for a
sev-1?" — and watch it call `search_corpus`.

**Type:** command-line tool (`local-rag-ingest`) and a stdio MCP server
(`local-rag-server`, the one run command for serving).

## Settings

All settings are environment variables; `.env.example` lists them.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | **required** | Ollama base URL. Startup fails naming it if unset. |
| `RAG_EMBED_MODEL` | **required** | Embedding model name. Startup fails naming it if unset. |
| `RAG_CORPUS_DIR` | `./sample-corpus` | Directory of `*.md` files to index |
| `RAG_STORE_DIR` | `~/.local-rag-mcp/store` | Local Chroma store location |

## Design decisions that earn their keep

- **The server is read-only and never creates stores.** Ingestion owns
  creation. A read-only server that quietly initializes an empty store turns
  "you forgot to ingest" into "search returns nothing" — the worse failure,
  because it looks like an answer.
- **Embed-then-swap ingest.** A file's old chunks are deleted only after
  every new chunk embedded successfully; an Ollama failure mid-file never
  leaves that file missing from the index.
- **Retired documents are pre-filtered, not post-filtered.** A document with
  `lifecycle: superseded` in its frontmatter is excluded by a `where` clause
  *before* the vector search, so it never occupies a result slot. Ingest
  writes the lifecycle key explicitly on every chunk — on some versions of
  the store a missing key slips through `$ne`, so absence is not a safe
  default. (The original of this rule exists because a re-ingest once
  silently erased the marker and a retired document resurfaced in results;
  a regression test now pins it.)
- **`get_file` is symlink-hardened.** Only indexed paths are readable, and a
  path that resolves somewhere different than it did at ingest time is
  refused — otherwise anyone who can swap a corpus file for a symlink reads
  outside the corpus through the server. If the file is absent on disk
  (moved corpus, different machine), the indexed chunk text is served
  instead, in chunk order.
- **The store is machine-local, always.** It's a live SQLite-backed
  database; cloud sync does whole-file replacement with no transactional
  awareness, and the failure mode is a silently corrupted index on the
  machine that didn't write it. Sync the corpus and this recipe; every
  machine builds its own store.
- **Every ingest stamps the corpus git commit** into its output, so an index
  build can be pinned to exactly the corpus state that produced it
  ("uncommitted changes present" is itself a warning label).
- **Asymmetric embedding prefixes** (EmbeddingGemma's documented query/doc
  instruction prefixes) on both sides of the search, matching the companion
  project's measured regime — prefixed beat raw retrieval by double digits
  there, and mixed prefixed/raw vectors score in an uncalibrated band.

## Corpus conventions

Any directory of `*.md` files works. Three optional frontmatter keys:

```yaml
rag: false            # exclude this file from the index entirely
rag_chunk: headings   # heading-split a long document (default: whole-file)
lifecycle: superseded # keep the file, hide it from search
```

The committed `sample-corpus/` exercises all three plus a plain file — six
fictional platform-team documents, generated by `tools/gen_sample_corpus.py`
(CI verifies the committed corpus matches the generator).

## Tests

```bash
pip install -e ".[dev]" && python -m pytest -q
```

No Ollama, no store: the embedder is stubbed (and `embed` itself is tested
against a fake loopback HTTP server) and the collection is a fake
that records calls. Under test are the contracts — argument validation, the
lifecycle pre-filter reaching the store as a `where` clause, the read-only
no-create guarantee, symlink refusal, embed-then-swap ordering (including
the embedder-down path), the mtime tolerance skip, and the chunker's merge
and oversize-split behavior.

## Known limitations

- **Trust model:** the server reads whatever corpus you point it at, and
  clients inject retrieved text into model context. Index only content you
  trust — a hostile document is a prompt-injection vector; the server
  retrieves, it doesn't sanitize. Stdio MCP has no authentication layer;
  it inherits the trust of the process that launched it.
- Scores are comparable only within one embedding regime; a calibrated
  "weak match" floor is corpus-specific (the companion repo documents the
  calibration method).
- Embedding prefixes are EmbeddingGemma's; `RAG_EMBED_MODEL` must name a
  model that uses them. Changing the model needs `local-rag-ingest --force`.
- Whole-file mode embeds only the first 6000 characters of a file; use
  `rag_chunk: headings` for long documents.
- Document identity is the absolute corpus path, so a store is tied to one
  machine's layout.
- One store, one collection — multi-corpus routing is out of scope here.
- No hybrid keyword+vector stage; paraphrase headroom is measured and
  documented in the companion repo.

## License

MIT — see [LICENSE](LICENSE).

## Author

Wes Glockzin
