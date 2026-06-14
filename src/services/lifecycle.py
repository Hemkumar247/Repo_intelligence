"""Repository indexing lifecycle and job tracking."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RepoLifecycleManager:
    def __init__(self, state_dir: str):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "lifecycle.json"
        self._lock = threading.Lock()
        self._state = self._load_state()

    def create_job(self, repo_url: str) -> Dict[str, Any]:
        with self._lock:
            job_id = uuid.uuid4().hex
            job = {
                "job_id": job_id,
                "repo_url": repo_url,
                "status": "queued",
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "result": None,
                "attempts": 0,
            }
            self._state["jobs"][job_id] = job
            self._save_state()
            return dict(job)

    def mark_running(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._require_job(job_id)
            job["status"] = "running"
            job["started_at"] = job["started_at"] or utc_now_iso()
            job["updated_at"] = utc_now_iso()
            job["attempts"] += 1
            self._save_state()
            return dict(job)

    def mark_completed(self, job_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            job = self._require_job(job_id)
            job["status"] = "completed"
            job["result"] = result
            job["finished_at"] = utc_now_iso()
            job["updated_at"] = utc_now_iso()
            repo_name = result.get("repo_name")
            if repo_name:
                self._state["repos"][repo_name] = {
                    "repo_name": repo_name,
                    "repo_url": job["repo_url"],
                    "last_indexed_at": job["finished_at"],
                    "last_job_id": job_id,
                    "status": "indexed",
                    "summary": {
                        "total_files": result.get("total_files"),
                        "total_chunks": result.get("total_chunks"),
                    },
                }
            self._save_state()
            return dict(job)

    def mark_failed(self, job_id: str, error: str) -> Dict[str, Any]:
        with self._lock:
            job = self._require_job(job_id)
            job["status"] = "failed"
            job["error"] = error
            job["finished_at"] = utc_now_iso()
            job["updated_at"] = utc_now_iso()
            self._save_state()
            return dict(job)

    def record_deleted_repo(self, repo_name: str) -> None:
        with self._lock:
            if repo_name in self._state["repos"]:
                self._state["repos"][repo_name]["status"] = "deleted"
                self._state["repos"][repo_name]["deleted_at"] = utc_now_iso()
            self._save_state()

    def list_jobs(self) -> Dict[str, Any]:
        with self._lock:
            jobs = sorted(
                self._state["jobs"].values(),
                key=lambda item: item["created_at"],
                reverse=True
            )
            return {"jobs": jobs}

    def list_repos(self) -> Dict[str, Any]:
        with self._lock:
            repos = sorted(
                self._state["repos"].values(),
                key=lambda item: item.get("last_indexed_at") or "",
                reverse=True
            )
            return {"repositories": repos}

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._require_job(job_id))

    def prepare_retry(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            previous = dict(self._require_job(job_id))
        return self.create_job(previous["repo_url"])

    def _require_job(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self._state["jobs"]:
            raise KeyError(f"Unknown job_id: {job_id}")
        return self._state["jobs"][job_id]

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_file.exists():
            return {"jobs": {}, "repos": {}}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return {"jobs": {}, "repos": {}}

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
