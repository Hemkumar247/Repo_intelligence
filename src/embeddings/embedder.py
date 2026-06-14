"""
Code Embeddings Module
Uses SentenceTransformers for semantic code understanding and chunking.
"""
import numpy as np
from typing import List, Dict, Optional
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
        """Split code into character-bounded chunks while keeping line references."""
        chunk_size = chunk_size or self.settings.chunk_size
        overlap = self.settings.chunk_overlap if overlap is None else overlap

        chunks = []
        lines = content.split("\n")
        current_chunk = []
        current_start_line = 1
        current_length = 0
        flush_threshold = max(chunk_size // 3, 256)

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
            current_length += len(line) + 1

            should_flush = current_length >= chunk_size
            blank_boundary = stripped == "" and len(current_chunk) > 5 and current_length >= flush_threshold
            if should_flush or blank_boundary:
                chunks.append(self._build_chunk(
                    current_chunk=current_chunk,
                    start_line=current_start_line,
                    end_line=i + 1,
                    context=context,
                    language=language
                ))

                current_chunk, current_start_line = self._trim_overlap(current_chunk, i + 1, overlap)
                current_length = sum(len(item) + 1 for item in current_chunk)

        # Add remaining chunk
        if current_chunk:
            chunks.append(self._build_chunk(
                current_chunk=current_chunk,
                start_line=current_start_line,
                end_line=len(lines),
                context=context,
                language=language
            ))

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

    def _build_chunk(
        self,
        current_chunk: List[str],
        start_line: int,
        end_line: int,
        context: Dict,
        language: str
    ) -> Dict:
        chunk_text = "\n".join(current_chunk)
        return {
            "text": chunk_text,
            "start_line": start_line,
            "end_line": end_line,
            "context": context.copy(),
            "type": self._detect_chunk_type(chunk_text, language)
        }

    def _trim_overlap(self, current_chunk: List[str], end_line: int, overlap: int) -> tuple[List[str], int]:
        if overlap <= 0 or not current_chunk:
            return [], end_line + 1

        overlap_lines: List[str] = []
        overlap_chars = 0
        for line in reversed(current_chunk):
            line_length = len(line) + 1
            if overlap_lines and overlap_chars + line_length > overlap:
                break
            overlap_lines.append(line)
            overlap_chars += line_length
            if overlap_chars >= overlap:
                break

        overlap_lines.reverse()
        overlap_start_line = end_line - len(overlap_lines) + 1
        return overlap_lines, overlap_start_line


class HybridEmbedder:
    """Uses a single sentence transformer model for both code and NL queries."""

    def __init__(self):
        self.code_embedder = CodeEmbedder()

    def embed(self, text: str, is_code: bool = False) -> np.ndarray:
        return self.code_embedder.embed_code(text)
