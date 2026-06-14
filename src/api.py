"""
FastAPI REST API
Provides programmatic access to the Repository Intelligence System.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from config.settings import get_settings
from src.services.runtime import get_lifecycle_manager, get_pipeline

app = FastAPI(
    title="GitHub Repo Intelligence API",
    description="AI-powered code analysis and understanding",
    version="1.1.0"
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def pipeline_dependency():
    return get_pipeline()


def lifecycle_dependency():
    return get_lifecycle_manager()


class IndexRequest(BaseModel):
    repo_url: HttpUrl


class IndexResponse(BaseModel):
    repo_name: str
    total_files: int
    total_chunks: int
    status: str


class IndexJobResponse(BaseModel):
    job_id: str
    repo_url: str
    status: str
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    attempts: int = 0


class QuestionRequest(BaseModel):
    repo_name: str
    question: str


class SourceReference(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    chunk_type: Optional[str] = None
    language: Optional[str] = None
    score: Optional[float] = None


class ContextPreview(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    snippet: str


class QuestionResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    confidence: str = "low"
    context_preview: List[ContextPreview] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    repo_name: str
    target: str
    target_type: str = "function"


class BugRequest(BaseModel):
    repo_name: str
    file_path: Optional[str] = None


class ArchitectureResponse(BaseModel):
    analysis: Dict[str, str]
    diagrams: Dict[str, str]
    metrics: Dict[str, Any]


def _run_index_job(job_id: str, repo_url: str) -> None:
    lifecycle = get_lifecycle_manager()
    pipeline = get_pipeline()
    lifecycle.mark_running(job_id)
    try:
        result = pipeline.index_repository(repo_url)
        lifecycle.mark_completed(job_id, result)
    except Exception as exc:
        lifecycle.mark_failed(job_id, str(exc))


@app.post("/index", response_model=IndexResponse)
async def index_repository(
    request: IndexRequest,
    pipeline=Depends(pipeline_dependency),
    lifecycle=Depends(lifecycle_dependency),
):
    """Synchronously index a GitHub repository."""
    try:
        result = pipeline.index_repository(str(request.repo_url))
        sync_job = lifecycle.create_job(str(request.repo_url))
        lifecycle.mark_running(sync_job["job_id"])
        lifecycle.mark_completed(sync_job["job_id"], result)
        return IndexResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/index/jobs", response_model=IndexJobResponse)
async def create_index_job(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    lifecycle=Depends(lifecycle_dependency),
):
    """Queue a background indexing job."""
    job = lifecycle.create_job(str(request.repo_url))
    background_tasks.add_task(_run_index_job, job["job_id"], str(request.repo_url))
    return IndexJobResponse(**job)


@app.get("/index/jobs", response_model=Dict[str, List[IndexJobResponse]])
async def list_index_jobs(lifecycle=Depends(lifecycle_dependency)):
    """List indexing jobs."""
    jobs = lifecycle.list_jobs()
    return {"jobs": [IndexJobResponse(**job) for job in jobs["jobs"]]}


@app.get("/index/jobs/{job_id}", response_model=IndexJobResponse)
async def get_index_job(job_id: str, lifecycle=Depends(lifecycle_dependency)):
    """Get indexing job status."""
    try:
        return IndexJobResponse(**lifecycle.get_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/index/jobs/{job_id}/retry", response_model=IndexJobResponse)
async def retry_index_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    lifecycle=Depends(lifecycle_dependency),
):
    """Retry a previous indexing job."""
    try:
        retry_job = lifecycle.prepare_retry(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    background_tasks.add_task(_run_index_job, retry_job["job_id"], retry_job["repo_url"])
    return IndexJobResponse(**retry_job)


@app.get("/repos")
async def list_repositories(lifecycle=Depends(lifecycle_dependency)):
    """List repositories known to the lifecycle registry."""
    return lifecycle.list_repos()


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, pipeline=Depends(pipeline_dependency)):
    """Ask a question about an indexed repository."""
    try:
        result = pipeline.ask_with_sources(request.repo_name, request.question)
        return QuestionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/explain")
async def explain_code(request: ExplainRequest, pipeline=Depends(pipeline_dependency)):
    """Explain a function or file."""
    try:
        if request.target_type == "function":
            result = pipeline.explain_function(request.repo_name, request.target)
        else:
            result = pipeline.explain_file(request.repo_name, request.target)
        return {"explanation": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/bugs")
async def find_bugs(request: BugRequest, pipeline=Depends(pipeline_dependency)):
    """Find bugs in the repository."""
    try:
        bugs = pipeline.find_bugs(request.repo_name, request.file_path)
        return {"bugs": bugs}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/architecture", response_model=ArchitectureResponse)
async def generate_architecture(request: QuestionRequest, pipeline=Depends(pipeline_dependency)):
    """Generate architecture analysis and diagrams."""
    try:
        result = pipeline.generate_architecture(request.repo_name)
        return ArchitectureResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/stats/{repo_name}")
async def get_stats(repo_name: str, pipeline=Depends(pipeline_dependency)):
    """Get repository statistics."""
    try:
        return pipeline.get_repo_stats(repo_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/repos/{repo_name}")
async def delete_repo(
    repo_name: str,
    pipeline=Depends(pipeline_dependency),
    lifecycle=Depends(lifecycle_dependency),
):
    """Delete a repository from the index."""
    try:
        pipeline.delete_repo(repo_name)
        lifecycle.record_deleted_repo(repo_name)
        return {"message": f"Repository {repo_name} deleted"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.1.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
