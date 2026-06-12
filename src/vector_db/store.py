"""
Vector Database Module
Manages code embeddings storage and retrieval with Qdrant.
"""
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import asdict
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue, ScoredPoint
)

from config.settings import get_settings

class VectorStore:
    def __init__(self):
        self.settings = get_settings()
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key
        )
        self.collection_name = self.settings.collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="language",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_type",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="repo_name",
                field_schema="keyword"
            )

    def index_chunks(self, chunks: List[Dict], repo_name: str, file_path: str) -> List[str]:
        """Index code chunks into vector database."""
        points = []
        ids = []

        for chunk in chunks:
            point_id = str(uuid.uuid4())
            ids.append(point_id)

            point = PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload={
                    "repo_name": repo_name,
                    "file_path": file_path,
                    "content": chunk["text"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "language": chunk.get("language", "unknown"),
                    "chunk_type": chunk.get("type", "code"),
                    "context": chunk.get("context", {})
                }
            )
            points.append(point)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i + batch_size]
            )

        return ids

    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """Search for similar code chunks."""
        search_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            search_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                "content": r.payload.get("content", ""),
                "file_path": r.payload.get("file_path", ""),
                "start_line": r.payload.get("start_line", 0),
                "end_line": r.payload.get("end_line", 0),
                "language": r.payload.get("language", ""),
                "chunk_type": r.payload.get("chunk_type", ""),
                "context": r.payload.get("context", {})
            }
            for r in results
        ]

    def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """Hybrid search combining vector similarity with keyword matching."""
        # Get vector search results
        vector_results = self.search(query_vector, top_k * 2, filters)

        # Boost scores for keyword matches in content
        query_terms = query_text.lower().split()
        for result in vector_results:
            content_lower = result["content"].lower()
            keyword_matches = sum(1 for term in query_terms if term in content_lower)
            # Combine scores: 70% vector, 30% keyword
            result["score"] = result["score"] * 0.7 + (keyword_matches / len(query_terms)) * 0.3 if query_terms else result["score"]

        # Re-sort by combined score
        vector_results.sort(key=lambda x: x["score"], reverse=True)
        return vector_results[:top_k]

    def get_by_file(self, repo_name: str, file_path: str) -> List[Dict]:
        """Get all chunks for a specific file."""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
                FieldCondition(key="file_path", match=MatchValue(value=file_path))
            ]),
            limit=1000,
            with_payload=True
        )

        return [
            {
                "id": r.id,
                "content": r.payload.get("content", ""),
                "start_line": r.payload.get("start_line", 0),
                "end_line": r.payload.get("end_line", 0)
            }
            for r in results[0]
        ]

    def delete_repo(self, repo_name: str):
        """Delete all chunks for a repository."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[
                FieldCondition(key="repo_name", match=MatchValue(value=repo_name))
            ])
        )

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status
        }
