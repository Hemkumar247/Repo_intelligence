"""
Test Suite for Repository Intelligence System
"""
import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.github.connector import GitHubConnector, CodeFile
from src.embeddings.embedder import CodeEmbedder
from src.vector_db.store import VectorStore

class TestGitHubConnector:
    def test_get_language(self):
        connector = GitHubConnector()
        assert connector.get_language("test.py") == "python"
        assert connector.get_language("test.js") == "javascript"
        assert connector.get_language("test.ts") == "typescript"
        assert connector.get_language("test.go") == "go"
        assert connector.get_language("test.rs") == "rust"

    def test_extract_imports_python(self):
        connector = GitHubConnector()
        code = """
import os
import sys
from collections import defaultdict
from typing import List
"""
        imports = connector.extract_imports(code, "python")
        assert "os" in imports
        assert "sys" in imports
        assert "collections" in imports
        assert "typing" in imports

    def test_extract_functions_python(self):
        connector = GitHubConnector()
        code = """
def hello(name: str) -> str:
    return f"Hello {name}"

async def fetch_data(url: str) -> dict:
    return {}
"""
        functions = connector.extract_functions(code, "python")
        assert len(functions) == 2
        assert functions[0]["name"] == "hello"
        assert functions[1]["name"] == "fetch_data"

    def test_should_ignore(self):
        connector = GitHubConnector()
        assert connector.should_ignore("node_modules/package.json")
        assert connector.should_ignore("__pycache__/test.pyc")
        assert not connector.should_ignore("src/main.py")

class TestCodeEmbedder:
    @patch('src.embeddings.embedder.AutoTokenizer')
    @patch('src.embeddings.embedder.AutoModel')
    def test_chunk_code(self, mock_model, mock_tokenizer):
        embedder = CodeEmbedder.__new__(CodeEmbedder)
        embedder.settings = Mock()
        embedder.settings.chunk_size = 100
        embedder.settings.chunk_overlap = 20

        code = """
import os

def hello():
    print("Hello")

def world():
    print("World")
"""
        chunks = embedder.chunk_code(code, "python")
        assert len(chunks) > 0
        assert all("text" in c for c in chunks)

    @patch('src.embeddings.embedder.AutoTokenizer')
    @patch('src.embeddings.embedder.AutoModel')
    def test_detect_chunk_type(self, mock_model, mock_tokenizer):
        embedder = CodeEmbedder.__new__(CodeEmbedder)

        assert embedder._detect_chunk_type("import os\nimport sys", "python") == "imports"
        assert embedder._detect_chunk_type("class MyClass:", "python") == "class"
        assert embedder._detect_chunk_type("def my_func():", "python") == "function"
        assert embedder._detect_chunk_type("x = 1 + 2", "python") == "code"

class TestVectorStore:
    @patch('src.vector_db.store.QdrantClient')
    def test_ensure_collection(self, mock_client):
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        mock_instance.get_collections.return_value = Mock(collections=[])

        store = VectorStore.__new__(VectorStore)
        store.settings = Mock()
        store.settings.collection_name = "test"
        store.settings.embedding_dimension = 768
        store.settings.qdrant_url = "http://localhost:6333"
        store.settings.qdrant_api_key = None
        store.client = mock_instance

        store._ensure_collection()
        mock_instance.create_collection.assert_called_once()

class TestIntegration:
    def test_end_to_end_mock(self):
        """Integration test with mocked dependencies."""
        # This would test the full pipeline with mocks
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
