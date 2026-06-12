from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache

class Settings(BaseSettings):
    # GitHub
    github_token: Optional[str] = None

    # LLM — Google Gemini
    gemini_api_key: Optional[str] = None
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.1

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 128

    # Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "repo_intelligence_v2"

    # Indexing
    max_file_size_kb: int = 500
    supported_extensions: List[str] = [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
        ".java", ".cpp", ".c", ".h", ".rb", ".php", ".swift",
        ".kt", ".scala", ".md", ".json", ".yaml", ".yml", ".toml"
    ]
    ignore_patterns: List[str] = [
        "node_modules", "venv", ".git", "__pycache__", ".idea",
        "dist", "build", "*.min.js", "*.lock", ".env", "coverage"
    ]

    # Agents
    max_iterations: int = 10
    recursion_limit: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
