"""
Vector Database Module
Manages code embeddings storage and retrieval with Qdrant.
"""
from collections import Counter
from datetime import datetime, timezone
import uuid
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
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
        indexed_at = datetime.now(timezone.utc).isoformat()

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
                    "context": chunk.get("context", {}),
                    "indexed_at": indexed_at
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
            scroll_filter=self._build_filter({"repo_name": repo_name, "file_path": file_path}),
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
            points_selector=self._build_filter({"repo_name": repo_name})
        )

    def get_stats(self, repo_name: Optional[str] = None) -> Dict:
        """Get collection or repository statistics."""
        info = self.client.get_collection(self.collection_name)
        if repo_name is None:
            return {
                "scope": "collection",
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }

        records = self._scroll_all({"repo_name": repo_name})
        payloads = [item.payload or {} for item in records]
        languages = Counter(payload.get("language", "unknown") for payload in payloads)
        chunk_types = Counter(payload.get("chunk_type", "code") for payload in payloads)
        file_paths = sorted({payload.get("file_path", "") for payload in payloads if payload.get("file_path")})
        indexed_times = sorted(payload.get("indexed_at") for payload in payloads if payload.get("indexed_at"))

        return {
            "scope": "repository",
            "repo_name": repo_name,
            "points_count": len(payloads),
            "files_count": len(file_paths),
            "file_paths": file_paths,
            "languages": dict(languages),
            "chunk_types": dict(chunk_types),
            "first_indexed_at": indexed_times[0] if indexed_times else None,
            "last_indexed_at": indexed_times[-1] if indexed_times else None,
            "collection_status": info.status
        }

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
        ]
        return Filter(must=conditions)

    def _scroll_all(self, filters: Optional[Dict[str, Any]] = None, limit: int = 256) -> List[Any]:
        records = []
        next_page = None
        scroll_filter = self._build_filter(filters) if filters else None

        while True:
            page, next_page = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                offset=next_page
            )
            records.extend(page)
            if next_page is None:
                break

        return records
