"""Guards the reason for the accepted chromadb advisories: this tool uses
Chroma only through PersistentClient and never starts or connects to a
Chroma server."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_NAMES = {"HttpClient", "AsyncHttpClient", "CloudClient"}
FORBIDDEN_MODULES = {"chromadb.server", "chromadb.cli"}


def _sources():
    files = [ROOT / n for n in ("common.py", "ingest.py", "server.py")]
    files += sorted((ROOT / "tools").glob("*.py"))
    return files


def _is_forbidden_module(name):
    return any(name == m or name.startswith(m + ".") for m in FORBIDDEN_MODULES)


def find_problems(source, label="<src>"):
    problems = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            problems.append(f"{label}:{node.lineno} uses {node.attr}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"{label}:{node.lineno} uses {node.id}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _is_forbidden_module(mod):
                problems.append(f"{label}:{node.lineno} imports {mod}")
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES:
                    problems.append(f"{label}:{node.lineno} imports {alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    problems.append(f"{label}:{node.lineno} imports {alias.name}")
    return problems


def test_code_never_uses_a_chroma_server():
    problems = []
    for path in _sources():
        problems += find_problems(path.read_text(encoding="utf-8"), path.name)
    assert not problems, "Chroma server use found: " + "; ".join(problems)


def test_detector_catches_server_use():
    assert find_problems("import chromadb\nchromadb.HttpClient()")
    assert find_problems("from chromadb import HttpClient")
    assert find_problems("import chromadb.server")
    assert not find_problems("import chromadb\nchromadb.PersistentClient(path='x')")


def test_persistent_client_is_still_the_way_in():
    text = (ROOT / "server.py").read_text(encoding="utf-8")
    assert "PersistentClient" in text
