"""
Code Embeddings Module
Uses SentenceTransformers for semantic code understanding and chunking.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import re

from config.settings import get_settings

class CodeEmbedder:
    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model

        # Load sentence transformer for both code and natural language queries
        self.model = SentenceTransformer(self.model_name)

    def chunk_code(self, content: str, language: str, chunk_size: int = None, overlap: int = None) -> List[Dict]:
        """Split code into semantic chunks preserving context."""
        chunk_size = chunk_size or self.settings.chunk_size
        overlap = overlap or self.settings.overlap_size if hasattr(self.settings, 'overlap_size') else 128

        chunks = []
        lines = content.split("\n")
        current_chunk = []
        current_length = 0

        # Track context: imports, class, function
        context = {"imports": [], "class": None, "function": None}

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Update context
            if language == "python":
                if stripped.startswith("import ") or stripped.startswith("from "):
                    context["imports"].append(stripped)
                elif stripped.startswith("class "):
                    context["class"] = stripped
                    context["function"] = None
                elif stripped.startswith("def ") or stripped.startswith("async def "):
                    context["function"] = stripped

            current_chunk.append(line)
            current_length += len(line)

            # Chunk boundary: function end, class end, or size limit
            if current_length >= chunk_size or (stripped == "" and len(current_chunk) > 5):
                chunk_text = "\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_line": i - len(current_chunk) + 1,
                    "end_line": i,
                    "context": context.copy(),
                    "type": self._detect_chunk_type(chunk_text, language)
                })

                # Overlap for continuity
                overlap_lines = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_lines.copy()
                current_length = sum(len(l) for l in current_chunk)

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start_line": len(lines) - len(current_chunk),
                "end_line": len(lines),
                "context": context.copy(),
                "type": self._detect_chunk_type(chunk_text, language)
            })

        return chunks

    def _detect_chunk_type(self, text: str, language: str) -> str:
        """Detect if chunk is function, class, import, or general code."""
        if language == "python":
            if re.search(r"^(import|from)\s+", text, re.M):
                return "imports"
            elif re.search(r"^class\s+", text, re.M):
                return "class"
            elif re.search(r"^(async\s+)?def\s+", text, re.M):
                return "function"
        return "code"

    def embed_code(self, code_text: str) -> np.ndarray:
        """Generate embedding for code snippet."""
        return self.model.encode(code_text)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed natural language query."""
        return self.model.encode(query)

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Embed all code chunks and add embeddings to metadata."""
        texts = [chunk["text"] for chunk in chunks]
        
        # SentenceTransformers encode can take a list of strings
        embeddings = self.model.encode(texts, batch_size=32)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()

        return chunks

    def compute_similarity(self, query_embedding: np.ndarray, code_embeddings: List[np.ndarray]) -> List[float]:
        """Compute cosine similarity between query and code embeddings."""
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        similarities = []
        for emb in code_embeddings:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sim = np.dot(query_norm, emb_norm)
            similarities.append(float(sim))

        return similarities


class HybridEmbedder:
    """Uses a single sentence transformer model for both code and NL queries."""

    def __init__(self):
        self.code_embedder = CodeEmbedder()

    def embed(self, text: str, is_code: bool = False) -> np.ndarray:
        return self.code_embedder.embed_code(text)
