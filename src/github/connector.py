"""
GitHub Repository Connector
Handles cloning, file extraction, and metadata collection.
"""
import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from github import Github
from git import Repo
import fnmatch

from config.settings import get_settings

@dataclass
class CodeFile:
    path: str
    content: str
    language: str
    size: int
    imports: List[str]
    functions: List[Dict]
    classes: List[Dict]

@dataclass
class RepoMetadata:
    name: str
    owner: str
    description: str
    stars: int
    language: str
    topics: List[str]
    default_branch: str
    total_files: int
    total_lines: int

class GitHubConnector:
    def __init__(self, token: Optional[str] = None):
        self.settings = get_settings()
        self.token = token or self.settings.github_token
        self.github = Github(self.token)
        self.temp_dir = None

    def clone_repo(self, repo_url: str) -> str:
        """Download a GitHub repository as a zip via the API (no git binary needed)."""
        import urllib.request
        import zipfile
        import io

        # Extract owner/repo from URL
        match = re.match(r"https?://github\.com/([^/]+)/([^/.]+)", repo_url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: {repo_url}")

        owner, repo_name = match.groups()
        repo_name = repo_name.replace(".git", "")

        # Use GitHub API to get default branch
        try:
            gh_repo = self.github.get_repo(f"{owner}/{repo_name}")
            branch = gh_repo.default_branch
        except Exception:
            branch = "main"

        # Download zip archive of the default branch
        zip_url = f"https://github.com/{owner}/{repo_name}/archive/refs/heads/{branch}.zip"
        headers = {}
        if self.token:
            headers["Authorization"] = f"token {self.token}"

        print(f"  >> Downloading {owner}/{repo_name} (branch: {branch})...")
        req = urllib.request.Request(zip_url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response:
            zip_data = response.read()

        # Extract to temp directory
        self.temp_dir = tempfile.mkdtemp(prefix="repo_intel_")
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            zf.extractall(self.temp_dir)

        # The zip extracts to a folder named <repo>-<branch>/
        extracted = [
            d for d in os.listdir(self.temp_dir)
            if os.path.isdir(os.path.join(self.temp_dir, d))
        ]
        if not extracted:
            raise RuntimeError("Failed to extract repository zip")

        clone_path = os.path.join(self.temp_dir, extracted[0])
        print(f"  OK  Downloaded and extracted to temp folder")
        return clone_path


    def get_repo_metadata(self, owner: str, repo_name: str) -> RepoMetadata:
        """Fetch repository metadata from GitHub API."""
        repo = self.github.get_repo(f"{owner}/{repo_name}")
        return RepoMetadata(
            name=repo.name,
            owner=repo.owner.login,
            description=repo.description or "",
            stars=repo.stargazers_count,
            language=repo.language or "Unknown",
            topics=repo.topics,
            default_branch=repo.default_branch,
            total_files=0,  # Will be populated during indexing
            total_lines=0
        )

    def should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored based on patterns."""
        settings = get_settings()
        for pattern in settings.ignore_patterns:
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern):
                return True
            if pattern in file_path.split(os.sep):
                return True
        return False

    def get_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "jsx", ".tsx": "tsx", ".go": "go", ".rs": "rust",
            ".java": "java", ".cpp": "cpp", ".c": "c", ".h": "c",
            ".rb": "ruby", ".php": "php", ".swift": "swift",
            ".kt": "kotlin", ".scala": "scala", ".md": "markdown",
            ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml"
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, "unknown")

    def extract_imports(self, content: str, language: str) -> List[str]:
        """Extract import statements from code."""
        imports = []

        if language == "python":
            # Match: import x, from x import y
            patterns = [
                r"^import\s+([\w.]+)",
                r"^from\s+([\w.]+)\s+import"
            ]
        elif language in ["javascript", "typescript", "jsx", "tsx"]:
            patterns = [
                r"""import\s+.*?\s+from\s+['"]([^'"]+)['"]""",
                r"""require\(['"]([^'"]+)['"]\)"""
            ]
        elif language == "go":
            patterns = [r'import\s+["`]([^"`]+)["`]']
        elif language == "rust":
            patterns = [r"use\s+([\w:]+)", r"extern\s+crate\s+(\w+)"]
        else:
            return imports

        for pattern in patterns:
            imports.extend(re.findall(pattern, content, re.MULTILINE))

        return list(set(imports))

    def extract_functions(self, content: str, language: str) -> List[Dict]:
        """Extract function definitions with signatures."""
        functions = []

        if language == "python":
            pattern = r"(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:"
            for match in re.finditer(pattern, content):
                functions.append({
                    "name": match.group(1),
                    "signature": match.group(0),
                    "params": match.group(2),
                    "return_type": match.group(3),
                    "line": content[:match.start()].count("\n") + 1
                })
        elif language in ["javascript", "typescript", "go", "rust"]:
            # Simplified patterns for other languages
            pattern = r"(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"
            for match in re.finditer(pattern, content):
                functions.append({
                    "name": match.group(1),
                    "signature": match.group(0),
                    "params": match.group(2),
                    "line": content[:match.start()].count("\n") + 1
                })

        return functions

    def extract_classes(self, content: str, language: str) -> List[Dict]:
        """Extract class definitions."""
        classes = []

        if language == "python":
            pattern = r"class\s+(\w+)(?:\(([^)]*)\))?:"
        elif language in ["javascript", "typescript"]:
            pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?"
        elif language == "java":
            pattern = r"(?:public\s+|private\s+|protected\s+)?class\s+(\w+)"
        elif language == "go":
            pattern = r"type\s+(\w+)\s+struct"
        elif language == "rust":
            pattern = r"(?:pub\s+)?struct\s+(\w+)"
        else:
            return classes

        for match in re.finditer(pattern, content):
            classes.append({
                "name": match.group(1),
                "signature": match.group(0),
                "line": content[:match.start()].count("\n") + 1
            })

        return classes

    def index_repository(self, repo_path: str) -> Tuple[List[CodeFile], RepoMetadata]:
        """Index all code files in the repository."""
        settings = get_settings()
        code_files = []
        total_lines = 0

        # Get metadata from git config or infer from path
        repo_name = os.path.basename(repo_path)

        for root, dirs, files in os.walk(repo_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self.should_ignore(os.path.join(root, d))]

            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)

                if self.should_ignore(rel_path):
                    continue

                ext = Path(file).suffix.lower()
                if ext not in settings.supported_extensions:
                    continue

                # Check file size
                size = os.path.getsize(file_path)
                if size > settings.max_file_size_kb * 1024:
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue

                language = self.get_language(file)
                imports = self.extract_imports(content, language)
                functions = self.extract_functions(content, language)
                classes = self.extract_classes(content, language)

                code_files.append(CodeFile(
                    path=rel_path,
                    content=content,
                    language=language,
                    size=size,
                    imports=imports,
                    functions=functions,
                    classes=classes
                ))

                total_lines += content.count("\n") + 1

        metadata = RepoMetadata(
            name=repo_name,
            owner="unknown",
            description="",
            stars=0,
            language="",
            topics=[],
            default_branch="main",
            total_files=len(code_files),
            total_lines=total_lines
        )

        return code_files, metadata

    def cleanup(self):
        """Remove temporary cloned repository."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
