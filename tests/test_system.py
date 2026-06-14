"""
Test suite for the Repository Intelligence System.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage


def _install_qdrant_stub() -> None:
    if "qdrant_client" in sys.modules:
        return

    qdrant_module = types.ModuleType("qdrant_client")
    qdrant_models = types.ModuleType("qdrant_client.models")

    class QdrantClient:
        def __init__(self, *args, **kwargs):
            self.collections = []

        def get_collections(self):
            return SimpleNamespace(collections=[])

        def create_collection(self, *args, **kwargs):
            return None

        def create_payload_index(self, *args, **kwargs):
            return None

        def upsert(self, *args, **kwargs):
            return None

        def search(self, *args, **kwargs):
            return []

        def scroll(self, *args, **kwargs):
            return ([], None)

        def delete(self, *args, **kwargs):
            return None

        def get_collection(self, *args, **kwargs):
            return SimpleNamespace(
                vectors_count=0,
                indexed_vectors_count=0,
                points_count=0,
                status="green"
            )

    class Distance:
        COSINE = "cosine"

    class _Model:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    for name in ["VectorParams", "PointStruct", "Filter", "FieldCondition", "MatchValue", "ScoredPoint"]:
        setattr(qdrant_models, name, type(name, (_Model,), {}))

    qdrant_module.QdrantClient = QdrantClient
    qdrant_models.Distance = Distance

    sys.modules["qdrant_client"] = qdrant_module
    sys.modules["qdrant_client.models"] = qdrant_models


_install_qdrant_stub()

from src.agents.workflow import RepoIntelligenceAgent
from src.diagrams.generator import DiagramGenerator
from src.embeddings.embedder import CodeEmbedder
from src.github.connector import CodeFile, GitHubConnector, RepoMetadata
from src.pipeline import RepoIntelligencePipeline
from src.services.lifecycle import RepoLifecycleManager
from src.vector_db.store import VectorStore


def _build_connector() -> GitHubConnector:
    connector = GitHubConnector.__new__(GitHubConnector)
    connector.settings = SimpleNamespace()
    connector.temp_dir = None
    connector._tree_sitter_parsers = {}
    return connector


def _build_embedder(chunk_size: int = 80, chunk_overlap: int = 20) -> CodeEmbedder:
    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder.settings = SimpleNamespace(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return embedder


def test_python_ast_extraction_uses_real_symbols():
    connector = _build_connector()
    code = """
import os
from collections import defaultdict

class Service:
    def run(self, value: int) -> str:
        return str(value)

async def fetch_data(url: str) -> dict:
    return {"url": url}
"""

    imports = connector.extract_imports(code, "python")
    functions = connector.extract_functions(code, "python")
    classes = connector.extract_classes(code, "python")

    assert imports == ["collections", "os"]
    assert [item["name"] for item in functions] == ["run", "fetch_data"]
    assert functions[1]["return_type"] == "dict"
    assert [item["name"] for item in classes] == ["Service"]


def test_chunk_code_uses_chunk_overlap_setting_with_character_overlap():
    embedder = _build_embedder(chunk_size=40, chunk_overlap=12)
    code = "\n".join([
        "def alpha():",
        "    return 'alpha'",
        "",
        "def beta():",
        "    return 'beta'",
        "",
        "def gamma():",
        "    return 'gamma'",
    ])

    chunks = embedder.chunk_code(code, "python")

    assert len(chunks) >= 2
    assert chunks[0]["end_line"] >= chunks[1]["start_line"]
    assert "def beta():" in chunks[0]["text"]
    assert "def beta():" in chunks[1]["text"]


def test_diagram_generator_falls_back_to_svg_and_resets_state():
    tmp_dir = Path("diagrams")
    generator = DiagramGenerator(output_dir=str(tmp_dir))
    code_file = CodeFile(
        path="src/service.py",
        content="class Service:\n    pass\ndef run():\n    pass",
        language="python",
        size=42,
        imports=["os"],
        functions=[{"name": "run", "line": 3}],
        classes=[{"name": "Service", "line": 1}]
    )

    generator.parse_codebase([code_file])
    first_path = generator.generate_component_diagram("demo")
    generator.parse_codebase([code_file])

    assert first_path.endswith(".svg")
    assert len(generator.components) == 2
    assert "Fallback diagram summary" in tmp_dir.joinpath("demo_components.svg").read_text(encoding="utf-8")


def test_pipeline_index_repository_batches_embeddings_and_cleans_up():
    pipeline = RepoIntelligencePipeline.__new__(RepoIntelligencePipeline)
    code_file = CodeFile(
        path="src/main.py",
        content="def run():\n    return 1",
        language="python",
        size=24,
        imports=[],
        functions=[{"name": "run", "line": 1}],
        classes=[]
    )
    metadata = RepoMetadata(
        name="demo-repo",
        owner="owner",
        description="demo",
        stars=0,
        language="python",
        topics=[],
        default_branch="main",
        total_files=1,
        total_lines=2
    )

    cleanup_calls: list[str] = []
    indexed_chunks: list[dict] = []
    parsed_codebases: list[list[CodeFile]] = []

    pipeline.github = SimpleNamespace(
        clone_repo=lambda repo_url: "C:/tmp/demo",
        index_repository=lambda repo_path: ([code_file], metadata),
        cleanup=lambda: cleanup_calls.append("cleanup")
    )
    pipeline.embedder = SimpleNamespace(
        settings=SimpleNamespace(max_chunks_per_file=20),
        chunk_code=lambda content, language: [{
            "text": content,
            "start_line": 1,
            "end_line": 2,
            "context": {},
            "type": "function"
        }],
        model=SimpleNamespace(
            encode=lambda texts, **kwargs: [np.array([0.1, 0.2, 0.3]) for _ in texts]
        )
    )
    pipeline.vector_store = SimpleNamespace(
        index_chunks=lambda chunks, repo_name, file_path: indexed_chunks.append({
            "repo_name": repo_name,
            "file_path": file_path,
            "chunks": chunks
        }) or ["chunk-1"]
    )
    pipeline.diagram_generator = SimpleNamespace(
        parse_codebase=lambda files: parsed_codebases.append(files)
    )

    result = pipeline.index_repository("https://github.com/example/demo")

    assert result["repo_name"] == "demo-repo"
    assert result["total_chunks"] == 1
    assert cleanup_calls == ["cleanup"]
    assert indexed_chunks[0]["repo_name"] == "demo-repo"
    assert parsed_codebases[0][0].path == "src/main.py"


def test_pipeline_generate_architecture_aggregates_outputs():
    pipeline = RepoIntelligencePipeline.__new__(RepoIntelligencePipeline)
    pipeline.architecture_agent = SimpleNamespace(
        analyze=lambda repo_name: {
            "overview": "overview",
            "communication": "communication",
            "patterns": "patterns",
            "data_flow": "data_flow"
        }
    )
    pipeline.diagram_generator = SimpleNamespace(
        generate_component_diagram=lambda repo_name: "component.svg",
        generate_class_diagram=lambda repo_name: "class.svg",
        generate_mermaid=lambda repo_name: "graph TD\nA-->B",
        generate_dependency_matrix=lambda repo_name: {"total_components": 2}
    )

    result = pipeline.generate_architecture("demo")

    assert result["diagrams"]["component"] == "component.svg"
    assert result["analysis"]["patterns"] == "patterns"
    assert result["metrics"]["total_components"] == 2


def test_vector_store_returns_repo_scoped_stats():
    class FakePoint:
        def __init__(self, payload):
            self.payload = payload

    class FakeClient:
        def get_collection(self, collection_name):
            return SimpleNamespace(
                vectors_count=10,
                indexed_vectors_count=10,
                points_count=10,
                status="green"
            )

        def scroll(self, **kwargs):
            return ([
                FakePoint({
                    "repo_name": "demo",
                    "file_path": "src/a.py",
                    "language": "python",
                    "chunk_type": "function",
                    "indexed_at": "2026-06-14T12:00:00+00:00"
                }),
                FakePoint({
                    "repo_name": "demo",
                    "file_path": "src/b.py",
                    "language": "python",
                    "chunk_type": "class",
                    "indexed_at": "2026-06-14T12:05:00+00:00"
                }),
            ], None)

    store = VectorStore.__new__(VectorStore)
    store.client = FakeClient()
    store.collection_name = "test"

    stats = store.get_stats("demo")

    assert stats["scope"] == "repository"
    assert stats["repo_name"] == "demo"
    assert stats["points_count"] == 2
    assert stats["files_count"] == 2
    assert stats["languages"] == {"python": 2}
    assert stats["chunk_types"] == {"function": 1, "class": 1}
    assert stats["last_indexed_at"] == "2026-06-14T12:05:00+00:00"


def test_lifecycle_manager_tracks_jobs_and_repositories():
    manager = RepoLifecycleManager("diagrams/test-state")
    job = manager.create_job("https://github.com/example/demo")
    running = manager.mark_running(job["job_id"])
    completed = manager.mark_completed(job["job_id"], {
        "repo_name": "demo",
        "total_files": 4,
        "total_chunks": 9,
    })

    fetched = manager.get_job(job["job_id"])
    repos = manager.list_repos()["repositories"]

    assert running["status"] == "running"
    assert completed["status"] == "completed"
    assert fetched["result"]["repo_name"] == "demo"
    assert repos[0]["repo_name"] == "demo"
    assert repos[0]["summary"]["total_chunks"] == 9


def test_agent_retrieve_context_deduplicates_files():
    captured = {}

    class FakeVectorStore:
        def hybrid_search(self, query_vector, query_text, top_k, filters):
            captured["filters"] = filters
            captured["query_text"] = query_text
            return [
                {
                    "file_path": "src/a.py",
                    "content": "def a(): pass",
                    "start_line": 1,
                    "end_line": 1,
                    "chunk_type": "function",
                    "context": {}
                },
                {
                    "file_path": "src/a.py",
                    "content": "def a_helper(): pass",
                    "start_line": 2,
                    "end_line": 2,
                    "chunk_type": "function",
                    "context": {}
                },
                {
                    "file_path": "src/b.py",
                    "content": "def b(): pass",
                    "start_line": 1,
                    "end_line": 1,
                    "chunk_type": "function",
                    "context": {}
                },
            ]

    agent = RepoIntelligenceAgent.__new__(RepoIntelligenceAgent)
    agent.vector_store = FakeVectorStore()
    agent.embedder = SimpleNamespace(embed_query=lambda query: np.array([0.5, 0.5, 0.5]))

    state = {
        "messages": [HumanMessage(content="How does auth work?")],
        "query": "How does auth work?",
        "context": [],
        "analysis": "",
        "task_type": "question",
        "repo_name": "demo",
        "iteration_count": 0,
        "final_answer": ""
    }

    updated = agent._retrieve_context(state)

    assert captured["filters"] == {"repo_name": "demo"}
    assert captured["query_text"] == "How does auth work?"
    assert [item["file_path"] for item in updated["context"]] == ["src/a.py", "src/b.py"]


def test_agent_bug_contract_uses_structured_json():
    agent = RepoIntelligenceAgent.__new__(RepoIntelligenceAgent)
    agent._run_workflow = lambda query, repo_name: {
        "analysis": "Potential issue found",
        "context": [{
            "file_path": "src/main.py",
            "start_line": 10,
            "end_line": 20,
            "chunk_type": "function",
            "language": "python",
            "content": "def run(): pass"
        }]
    }
    agent._format_context = lambda context: "formatted context"
    agent.llm = SimpleNamespace(
        invoke=lambda messages: SimpleNamespace(content="""{
            "summary": "One likely bug found",
            "bugs": [{
                "title": "Missing validation",
                "severity": "medium",
                "file_path": "src/main.py",
                "start_line": 10,
                "end_line": 12,
                "description": "Input is used without validation.",
                "recommendation": "Validate input before use."
            }]
        }""")
    )

    result = agent.find_bugs_structured("Find bugs", "demo")

    assert result["summary"] == "One likely bug found"
    assert result["bugs"][0]["title"] == "Missing validation"
    assert result["sources"][0]["file_path"] == "src/main.py"


def test_api_routes_use_pipeline_contract(monkeypatch):
    fake_pipeline_module = types.ModuleType("src.pipeline")

    class FakePipeline:
        def index_repository(self, repo_url):
            return {"repo_name": "demo", "total_files": 3, "total_chunks": 8, "status": "indexed"}

        def ask(self, repo_name, question):
            return f"Answer for {repo_name}: {question}"

        def ask_with_sources(self, repo_name, question):
            return {
                "answer": f"Answer for {repo_name}: {question}",
                "confidence": "high",
                "sources": [{
                    "file_path": "src/main.py",
                    "start_line": 1,
                    "end_line": 5,
                    "chunk_type": "function",
                    "language": "python"
                }],
                "context_preview": [{
                    "file_path": "src/main.py",
                    "start_line": 1,
                    "end_line": 5,
                    "snippet": "def main(): pass"
                }],
            }

        def explain_function(self, repo_name, target):
            return f"Function {target}"

        def explain_file(self, repo_name, target):
            return f"File {target}"

        def find_bugs(self, repo_name, file_path=None):
            return [{"title": "Potential issue", "details": ["Check edge case"]}]

        def generate_architecture(self, repo_name):
            return {
                "analysis": {
                    "overview": "overview",
                    "communication": "communication",
                    "patterns": "patterns",
                    "data_flow": "data_flow"
                },
                "diagrams": {
                    "component": "component.svg",
                    "class": "class.svg",
                    "mermaid": "graph TD\nA-->B"
                },
                "metrics": {
                    "total_components": 2,
                    "total_dependencies": 1,
                    "average_degree": 1.0
                }
            }

        def get_repo_stats(self, repo_name):
            return {"repo_name": repo_name, "points_count": 8}

        def delete_repo(self, repo_name):
            return None

    fake_pipeline_module.RepoIntelligencePipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "src.pipeline", fake_pipeline_module)
    sys.modules.pop("src.api", None)

    api_module = importlib.import_module("src.api")
    client = TestClient(api_module.app)

    health = client.get("/health")
    ask = client.post("/ask", json={"repo_name": "demo", "question": "What is this?"})
    architecture = client.post("/architecture", json={"repo_name": "demo", "question": "ignored"})

    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert ask.status_code == 200
    assert ask.json()["answer"] == "Answer for demo: What is this?"
    assert ask.json()["confidence"] == "high"
    assert ask.json()["sources"][0]["file_path"] == "src/main.py"
    assert ask.json()["context_preview"][0]["snippet"] == "def main(): pass"
    assert architecture.status_code == 200
    assert architecture.json()["diagrams"]["component"] == "component.svg"
