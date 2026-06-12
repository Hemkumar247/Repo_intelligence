"""
FastAPI REST API
Provides programmatic access to the Repository Intelligence System.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import uvicorn

import sys
from pathlib import Path

# Add project root to sys.path so 'src' package is discoverable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline import RepoIntelligencePipeline

app = FastAPI(
    title="GitHub Repo Intelligence API",
    description="AI-powered code analysis and understanding",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = RepoIntelligencePipeline()

# Request/Response Models
class IndexRequest(BaseModel):
    repo_url: HttpUrl

class IndexResponse(BaseModel):
    repo_name: str
    total_files: int
    total_chunks: int
    status: str

class QuestionRequest(BaseModel):
    repo_name: str
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class ExplainRequest(BaseModel):
    repo_name: str
    target: str  # function name or file path
    target_type: str = "function"  # "function" or "file"

class BugRequest(BaseModel):
    repo_name: str
    file_path: Optional[str] = None

class ArchitectureResponse(BaseModel):
    analysis: Dict[str, str]
    diagrams: Dict[str, str]
    metrics: Dict[str, Any]

# API Endpoints
@app.post("/index", response_model=IndexResponse)
async def index_repository(request: IndexRequest, background_tasks: BackgroundTasks):
    """Index a GitHub repository."""
    try:
        result = pipeline.index_repository(str(request.repo_url))
        return IndexResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about an indexed repository."""
    try:
        answer = pipeline.ask(request.repo_name, request.question)
        return QuestionResponse(
            answer=answer,
            sources=[]  # Could include source references
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain")
async def explain_code(request: ExplainRequest):
    """Explain a function or file."""
    try:
        if request.target_type == "function":
            result = pipeline.explain_function(request.repo_name, request.target)
        else:
            result = pipeline.explain_file(request.repo_name, request.target)
        return {"explanation": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/bugs")
async def find_bugs(request: BugRequest):
    """Find bugs in the repository."""
    try:
        bugs = pipeline.find_bugs(request.repo_name, request.file_path)
        return {"bugs": bugs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/architecture", response_model=ArchitectureResponse)
async def generate_architecture(request: QuestionRequest):
    """Generate architecture analysis and diagrams."""
    try:
        result = pipeline.generate_architecture(request.repo_name)
        return ArchitectureResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats/{repo_name}")
async def get_stats(repo_name: str):
    """Get repository statistics."""
    try:
        stats = pipeline.get_repo_stats(repo_name)
        return stats
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/repos/{repo_name}")
async def delete_repo(repo_name: str):
    """Delete a repository from the index."""
    try:
        pipeline.delete_repo(repo_name)
        return {"message": f"Repository {repo_name} deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
