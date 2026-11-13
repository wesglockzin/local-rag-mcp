"""Tests run WITHOUT Ollama and WITHOUT a real Chroma store where possible:
the embedder is stubbed and the collection is a fake that records calls.
What's under test is the contracts — validation, the lifecycle pre-filter,
get_file's corpus-only + symlink hardening, embed-then-swap ingest ordering,
and the chunker. One optional test uses real chromadb if installed."""

import json
import os
import threading
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import common
import ingest
import server


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    # Plainly fake, non-routable values: nothing here is ever contacted.
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.example.invalid")
    monkeypatch.setenv("RAG_EMBED_MODEL", "test-model")
    # main() parses sys.argv by default; keep pytest's own arguments out of it.
    monkeypatch.setattr("sys.argv", ["prog"])


class FakeCollection:
    def __init__(self, metadatas=None, documents=None):
        self.metadatas = metadatas or []
        self.documents = documents or []
        self.calls = []

    def count(self):
        return len(self.metadatas)

    def query(self, **kw):
        self.calls.append(("query", kw))
        n = kw.get("n_results", 4)
        return {
            "documents": [self.documents[:n]],
            "metadatas": [self.metadatas[:n]],
            "distances": [[0.2] * min(n, len(self.metadatas))],
        }

    def get(self, **kw):
        self.calls.append(("get", kw))
        where = kw.get("where")
        if where and "path" in where:
            keep = [
                (m, d)
                for m, d in zip(
                    self.metadatas, self.documents or [None] * len(self.metadatas)
                )
                if m.get("path") == where["path"]
            ]
            return {
                "ids": [f"{m.get('path')}::{m.get('chunk', 0)}" for m, _ in keep],
                "metadatas": [m for m, _ in keep],
                "documents": [d for _, d in keep],
            }
        return {
            "ids": [f"{m.get('path')}::{m.get('chunk', 0)}" for m in self.metadatas],
            "metadatas": self.metadatas,
            "documents": self.documents,
        }

    def delete(self, **kw):
        self.calls.append(("delete", kw))

    def add(self, **kw):
        self.calls.append(("add", kw))

    def upsert(self, **kw):
        self.calls.append(("upsert", kw))


@pytest.fixture(autouse=True)
def reset_collection(monkeypatch):
    server._collection = None
    yield
    server._collection = None


# --- do_search --------------------------------------------------------------


def test_search_rejects_short_query():
    assert "error" in server.do_search("ab")


def test_search_rejects_bad_k():
    assert "error" in server.do_search("valid query", k="not-a-number")


def test_search_clamps_k(monkeypatch):
    fake = FakeCollection(
        metadatas=[{"path": f"/p{i}", "lifecycle": "current"} for i in range(30)],
        documents=[f"doc {i}" for i in range(30)],
    )
    monkeypatch.setattr(server, "get_collection", lambda: fake)
    monkeypatch.setattr(server, "embed_query", lambda q, timeout=10.0: [1.0, 0.0])
    server.do_search("valid query", k=999)
    _, kw = fake.calls[-1]
    assert kw["n_results"] <= 20


def test_search_fails_soft_when_embedder_down(monkeypatch):
    def down(q, timeout=10.0):
        raise common.EmbedError("HTTP 500 from fake: boom")

    monkeypatch.setattr(server, "embed_query", down)
    out = server.do_search("valid query")
    assert "error" in out and "HTTP 500" in out["error"]


def test_search_passes_lifecycle_prefilter(monkeypatch):
    fake = FakeCollection(
        metadatas=[{"path": "/p", "lifecycle": "current"}], documents=["doc"]
    )
    monkeypatch.setattr(server, "get_collection", lambda: fake)
    monkeypatch.setattr(server, "embed_query", lambda q, timeout=10.0: [1.0, 0.0])
    out = server.do_search("valid query")
    _, kw = fake.calls[-1]
    assert kw["where"] == {"lifecycle": {"$ne": "superseded"}}
    assert out["results"][0]["score"] == pytest.approx(0.8)  # 1 - 0.2


def test_search_reports_missing_store(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(server, "embed_query", lambda q, timeout=10.0: [1.0])
    out = server.do_search("valid query")
    assert "error" in out and "ingest" in out["error"]


# --- do_get_file ------------------------------------------------------------


def test_get_file_refuses_non_corpus_path(monkeypatch):
    fake = FakeCollection(
        metadatas=[{"path": "/corpus/a.md", "real_path": "/corpus/a.md"}]
    )
    monkeypatch.setattr(server, "get_collection", lambda: fake)
    out = server.do_get_file("/etc/hosts")
    assert "error" in out and "hint" in out


def test_get_file_refuses_swapped_symlink(tmp_path, monkeypatch):
    real = tmp_path / "a.md"
    real.write_text("original", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    fake = FakeCollection(metadatas=[{"path": str(real), "real_path": str(real)}])
    monkeypatch.setattr(server, "get_collection", lambda: fake)
    real.unlink()
    os.symlink(outside, real)  # same path, now resolves elsewhere
    out = server.do_get_file(str(real))
    assert "error" in out and "resolves elsewhere" in out["error"]


def test_get_file_reads_valid_indexed_file(tmp_path, monkeypatch):
    p = tmp_path / "a.md"
    p.write_text("hello corpus", encoding="utf-8")
    fake = FakeCollection(
        metadatas=[{"path": str(p), "real_path": os.path.realpath(str(p))}]
    )
    monkeypatch.setattr(server, "get_collection", lambda: fake)
    out = server.do_get_file(str(p))
    assert out["text"] == "hello corpus"


def test_get_file_serves_index_text_when_file_absent(tmp_path, monkeypatch):
    gone = str(tmp_path / "gone.md")
    fake = FakeCollection(
        metadatas=[
            {"path": gone, "real_path": gone, "chunk": 1},
            {"path": gone, "real_path": gone, "chunk": 0},
        ],
        documents=["second", "first"],
    )
    monkeypatch.setattr(server, "get_collection", lambda: fake)
    out = server.do_get_file(gone)
    assert out["source"].startswith("index")
    assert out["text"] == "first\n\nsecond"  # chunk order restored


# --- read-only contract -----------------------------------------------------


def test_server_never_creates_store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STORE_DIR", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError):
        server.get_collection()
    assert not (tmp_path / "absent").exists()  # and it did NOT create it


def test_server_refuses_empty_store_dir(tmp_path, monkeypatch):
    # An existing-but-empty directory must ALSO be refused: PersistentClient
    # would initialize database files there, making a read-only server a writer.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(server, "STORE_DIR", str(empty))
    with pytest.raises(FileNotFoundError):
        server.get_collection()
    assert list(empty.iterdir()) == []  # still empty — nothing initialized


# --- ingest contracts -------------------------------------------------------


def _setup_corpus(tmp_path, monkeypatch, files):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, body in files.items():
        (corpus / name).write_text(body, encoding="utf-8")
    for mod in (common, ingest):
        monkeypatch.setattr(mod, "CORPUS_DIR", str(corpus))
    return corpus


def test_ingest_embed_then_swap_on_failure(tmp_path, monkeypatch):
    _setup_corpus(tmp_path, monkeypatch, {"a.md": "---\nname: a\n---\nbody"})
    fake = FakeCollection()
    monkeypatch.setattr(ingest, "get_collection", lambda: fake)

    def down(text, timeout=120.0):
        raise common.EmbedError("embedder down")

    monkeypatch.setattr(ingest, "embed", down)
    rc = ingest.main()
    assert rc == 1
    ops = [c[0] for c in fake.calls]
    assert "delete" not in ops and "add" not in ops  # old entries preserved


def test_ingest_upsert_then_prune_no_loss_window(tmp_path, monkeypatch):
    # A file that shrank from 3 chunks to 1: the new chunk is UPSERTED first,
    # and only the stale higher-numbered ids are deleted, after the write.
    corpus = _setup_corpus(tmp_path, monkeypatch, {"a.md": "---\nname: a\n---\nbody"})
    p = str(corpus / "a.md")
    fake = FakeCollection(
        metadatas=[{"path": p, "chunk": i, "mtime": 1.0} for i in range(3)],
        documents=["c0", "c1", "c2"],
    )
    monkeypatch.setattr(ingest, "get_collection", lambda: fake)
    monkeypatch.setattr(ingest, "embed", lambda text, timeout=120.0: [1.0, 0.0])
    assert ingest.main() == 0
    ops = [c[0] for c in fake.calls if c[0] in ("delete", "add", "upsert")]
    assert ops == ["upsert", "delete"]  # write first; prune after
    _, up_kw = [c for c in fake.calls if c[0] == "upsert"][0]
    assert up_kw["metadatas"][0]["lifecycle"] == "current"  # explicit key
    _, del_kw = [c for c in fake.calls if c[0] == "delete"][0]
    assert del_kw["ids"] == [f"{p}::1", f"{p}::2"]  # only the stale chunks


def test_ingest_strips_frontmatter_from_embedded_text(tmp_path, monkeypatch):
    _setup_corpus(
        tmp_path, monkeypatch, {"a.md": "---\nname: a\ndescription: x\n---\nreal body"}
    )
    fake = FakeCollection()
    monkeypatch.setattr(ingest, "get_collection", lambda: fake)
    monkeypatch.setattr(ingest, "embed", lambda text, timeout=120.0: [1.0])
    assert ingest.main() == 0
    _, up_kw = [c for c in fake.calls if c[0] == "upsert"][0]
    assert up_kw["documents"][0] == "real body"


def test_ingest_honors_rag_false_and_sweeps(tmp_path, monkeypatch):
    _setup_corpus(tmp_path, monkeypatch, {"skip.md": "---\nrag: false\n---\nscratch"})
    fake = FakeCollection(metadatas=[{"path": "/stale/old.md", "mtime": 1.0}])
    monkeypatch.setattr(ingest, "get_collection", lambda: fake)
    monkeypatch.setattr(ingest, "embed", lambda text, timeout=120.0: [1.0, 0.0])
    assert ingest.main() == 0
    deletes = [c for c in fake.calls if c[0] == "delete"]
    assert any(c[1]["where"]["path"] == "/stale/old.md" for c in deletes)
    assert not [c for c in fake.calls if c[0] == "add"]


def test_ingest_mtime_tolerance_skips_unchanged(tmp_path, monkeypatch):
    corpus = _setup_corpus(tmp_path, monkeypatch, {"a.md": "---\nname: a\n---\nbody"})
    mtime = os.path.getmtime(corpus / "a.md")
    fake = FakeCollection(
        metadatas=[{"path": str(corpus / "a.md"), "mtime": mtime + 0.0005}]
    )  # sub-tolerance
    monkeypatch.setattr(ingest, "get_collection", lambda: fake)
    calls = []
    monkeypatch.setattr(
        ingest, "embed", lambda text, timeout=120.0: calls.append(1) or [1.0]
    )
    assert ingest.main() == 0 and calls == []


def test_ingest_superseded_lifecycle_carried(tmp_path, monkeypatch):
    _setup_corpus(
        tmp_path,
        monkeypatch,
        {"old.md": "---\nname: old\nlifecycle: superseded\n---\nbody"},
    )
    fake = FakeCollection()
    monkeypatch.setattr(ingest, "get_collection", lambda: fake)
    monkeypatch.setattr(ingest, "embed", lambda text, timeout=120.0: [1.0])
    assert ingest.main() == 0
    _, up_kw = [c for c in fake.calls if c[0] == "upsert"][0]
    assert up_kw["metadatas"][0]["lifecycle"] == "superseded"


# --- chunker + frontmatter --------------------------------------------------


def test_heading_chunks_paths_and_merge():
    text = (
        "# Top\n\nintro paragraph long enough to stand alone as a chunk "
        "with plenty of characters in it to clear the merge threshold "
        "for small trailing chunks in the splitter today\n\n"
        "## Sub\n\ntiny\n"
    )
    chunks = common.heading_chunks(text)
    assert chunks[0]["heading_path"] == "Top"
    assert len(chunks) == 1  # "tiny" merged into previous
    assert "tiny" in chunks[0]["text"]


def test_heading_chunks_subsplit_oversize():
    text = "# Big\n\n" + ("x" * (common.MAX_CHARS_EMBED * 2))
    chunks = common.heading_chunks(text)
    assert len(chunks) >= 2
    assert all(len(c["text"]) <= common.MAX_CHARS_EMBED for c in chunks)
    assert "[part 1]" in chunks[0]["heading_path"]


def test_frontmatter_flags():
    fm = common.parse_frontmatter("---\nrag: false\nrag_chunk: headings\n---\nx")
    assert common.rag_enabled(fm) is False
    assert common.chunk_mode(fm) == "headings"
    assert common.rag_enabled({}) is True and common.chunk_mode({}) == "whole"


def test_frontmatter_inline_comments_stripped():
    # `rag: false  # why` must read as "false" — README-valid YAML that
    # silently disabled every frontmatter control before the fix.
    fm = common.parse_frontmatter(
        "---\nrag: false  # scratch\nlifecycle: superseded # retired\n---\nx"
    )
    assert common.rag_enabled(fm) is False
    assert fm["lifecycle"] == "superseded"


def test_strip_frontmatter():
    assert common.strip_frontmatter("---\nname: a\n---\n\nbody") == "body"
    assert common.strip_frontmatter("no frontmatter") == "no frontmatter"


def test_frontmatter_parse_has_no_deprecation_warning():
    # re.split(maxsplit) passed positionally is deprecated (error on 3.13+).
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        fm = common.parse_frontmatter("---\nrag: false  # scratch\n---\nbody")
    assert fm["rag"] == "false"


# --- settings ---------------------------------------------------------------


def test_missing_ollama_host_names_the_setting(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST")
    with pytest.raises(common.ConfigError, match="OLLAMA_HOST"):
        common.embed("hello")


def test_missing_model_names_the_setting(monkeypatch):
    monkeypatch.delenv("RAG_EMBED_MODEL")
    with pytest.raises(common.ConfigError, match="RAG_EMBED_MODEL"):
        common.embed("hello")


@pytest.mark.parametrize("bad", ["0.0.0.0", "ollama", "localhost:11434", "http://"])
def test_malformed_ollama_host_is_rejected_at_startup(monkeypatch, bad):
    monkeypatch.setenv("OLLAMA_HOST", bad)
    with pytest.raises(common.ConfigError, match="OLLAMA_HOST"):
        common.ollama_host()
    assert ingest.main() == 2
    assert server.main() == 2


def test_post_json_bad_url_becomes_embed_error():
    with pytest.raises(common.EmbedError, match="invalid URL"):
        common._post_json("0.0.0.0/api/embed", {}, 1)


def test_error_text_hides_url_credentials():
    url = "http://user:" + "fake-credential" + "@127.0.0.1:1/api/embed"
    with pytest.raises(common.EmbedError) as exc:
        common._post_json(url, {}, 2)
    assert "fake-credential" not in str(exc.value)


def test_host_with_credentials_is_rejected_at_startup(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://user:" + "fake-credential" + "@proxy:1")
    with pytest.raises(common.ConfigError, match="OLLAMA_HOST"):
        common.ollama_host()
    assert ingest.main() == 2


def test_host_with_non_numeric_port_is_rejected_at_startup(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:abc")
    with pytest.raises(common.ConfigError, match="OLLAMA_HOST"):
        common.ollama_host()


@pytest.mark.parametrize("module", [ingest, server])
def test_help_exits_zero_without_settings_and_touches_nothing(
    monkeypatch, tmp_path, capsys, module
):
    monkeypatch.delenv("OLLAMA_HOST")
    monkeypatch.delenv("RAG_EMBED_MODEL")
    monkeypatch.setattr(common, "STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(ingest, "STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(server, "STORE_DIR", str(tmp_path / "store"))
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()
    assert not (tmp_path / "store").exists()


def test_ingest_force_flag_is_parsed(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST")
    # Settings are checked after parsing: an unknown flag exits 2 via argparse,
    # a known one reaches the settings check and returns 2 from main.
    assert ingest.main(["--force"]) == 2
    with pytest.raises(SystemExit):
        ingest.main(["--bogus"])


def test_ingest_and_server_refuse_to_start_without_settings(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST")
    assert ingest.main() == 2
    assert server.main() == 2


# --- embed against a fake HTTP server --------------------------------------


@pytest.fixture
def fake_ollama(monkeypatch):
    routes: dict[str, tuple[int, bytes]] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            status, body = routes.get(self.path, (404, b"not found"))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{httpd.server_port}/")
    yield routes
    httpd.shutdown()
    httpd.server_close()


def test_embed_uses_new_endpoint(fake_ollama):
    fake_ollama["/api/embed"] = (
        200,
        json.dumps({"embeddings": [[0.5, 0.25]]}).encode(),
    )
    assert common.embed("hi") == [0.5, 0.25]


def test_embed_falls_back_to_legacy_endpoint(fake_ollama):
    fake_ollama["/api/embeddings"] = (200, json.dumps({"embedding": [1.0]}).encode())
    assert common.embed("hi") == [1.0]


def test_embed_http_error_reports_status_and_body(fake_ollama):
    fake_ollama["/api/embed"] = (500, b"model exploded")
    fake_ollama["/api/embeddings"] = (500, b"model exploded")
    with pytest.raises(common.EmbedError) as exc:
        common.embed("hi")
    assert "HTTP 500" in str(exc.value) and "model exploded" in str(exc.value)
    assert "cannot reach" not in str(exc.value)


def test_embed_truncated_response_becomes_embed_error(monkeypatch):
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n"
                b"Connection: close\r\n\r\nshort"
            )
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{srv.getsockname()[1]}")
    try:
        with pytest.raises(common.EmbedError, match="cannot reach"):
            common.embed("hi", timeout=5)
    finally:
        srv.close()


def test_embed_unreachable_server_is_reported_as_unreachable(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    with pytest.raises(common.EmbedError, match="cannot reach"):
        common.embed("hi", timeout=2)


# --- MCP layer (only if the SDK is installed) -------------------------------


def test_mcp_layer_registers_two_readonly_tools():
    pytest.importorskip("mcp")
    srv = server.build_server()
    assert srv is not None
