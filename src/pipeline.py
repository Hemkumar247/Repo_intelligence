"""
Main Pipeline Orchestrator
Coordinates the entire repository intelligence workflow.
"""
import os
from typing import Optional, Dict, List
from dataclasses import asdict
import json

from src.github.connector import GitHubConnector, CodeFile, RepoMetadata
from src.embeddings.embedder import CodeEmbedder
from src.vector_db.store import VectorStore
from src.agents.workflow import RepoIntelligenceAgent, ArchitectureAgent, BugFinderAgent
from src.diagrams.generator import DiagramGenerator

class RepoIntelligencePipeline:
    """End-to-end pipeline for repository intelligence."""

    def __init__(self):
        self.github = GitHubConnector()
        self.embedder = CodeEmbedder()
        self.vector_store = VectorStore()
        self.agent = RepoIntelligenceAgent()
        self.architecture_agent = ArchitectureAgent(self.agent)
        self.bug_finder = BugFinderAgent(self.agent)
        self.diagram_generator = DiagramGenerator()

    def index_repository(self, repo_url: str) -> Dict:
        """Full indexing pipeline for a GitHub repository."""
        print(f"🔍 Cloning repository: {repo_url}")
        repo_path = self.github.clone_repo(repo_url)

        print("📁 Indexing files...")
        code_files, metadata = self.github.index_repository(repo_path)

        print(f"📊 Found {len(code_files)} code files")

        # Index each file
        total_chunks = 0
        for code_file in code_files:
            print(f"  📝 Processing {code_file.path}...")

            # Chunk the code
            chunks = self.embedder.chunk_code(
                content=code_file.content,
                language=code_file.language
            )

            # Add metadata to chunks
            for chunk in chunks:
                chunk["language"] = code_file.language

            # Embed chunks
            chunks = self.embedder.embed_chunks(chunks)

            # Store in vector DB
            chunk_ids = self.vector_store.index_chunks(
                chunks=chunks,
                repo_name=metadata.name,
                file_path=code_file.path
            )

            total_chunks += len(chunk_ids)

        # Parse for diagrams
        self.diagram_generator.parse_codebase(code_files)

        # Cleanup
        self.github.cleanup()

        return {
            "repo_name": metadata.name,
            "total_files": len(code_files),
            "total_chunks": total_chunks,
            "metadata": asdict(metadata),
            "status": "indexed"
        }

    def ask(self, repo_name: str, question: str) -> str:
        """Ask a question about an indexed repository."""
        return self.agent.ask(question, repo_name)

    def explain_function(self, repo_name: str, function_name: str) -> str:
        """Explain a specific function."""
        query = f"Explain the function '{function_name}' in detail. What does it do, what are its parameters, and what does it return?"
        return self.agent.ask(query, repo_name)

    def explain_file(self, repo_name: str, file_path: str) -> str:
        """Explain a specific file."""
        query = f"Explain the file '{file_path}'. What is its purpose, key components, and how does it fit into the architecture?"
        return self.agent.ask(query, repo_name)

    def find_bugs(self, repo_name: str, specific_file: Optional[str] = None) -> List[Dict]:
        """Find potential bugs in the repository."""
        return self.bug_finder.find_bugs(repo_name, specific_file)

    def generate_architecture(self, repo_name: str) -> Dict:
        """Generate architecture analysis and diagrams."""
        # Get text analysis
        analysis = self.architecture_agent.analyze(repo_name)

        # Generate diagrams
        component_diagram = self.diagram_generator.generate_component_diagram(repo_name)
        class_diagram = self.diagram_generator.generate_class_diagram(repo_name)
        mermaid = self.diagram_generator.generate_mermaid(repo_name)
        dependency_metrics = self.diagram_generator.generate_dependency_matrix(repo_name)

        return {
            "analysis": analysis,
            "diagrams": {
                "component": component_diagram,
                "class": class_diagram,
                "mermaid": mermaid
            },
            "metrics": dependency_metrics
        }

    def get_repo_stats(self, repo_name: str) -> Dict:
        """Get repository statistics."""
        return self.vector_store.get_stats()

    def delete_repo(self, repo_name: str):
        """Remove a repository from the index."""
        self.vector_store.delete_repo(repo_name)


# CLI Interface
class CLI:
    """Command-line interface for the system."""

    def __init__(self):
        self.pipeline = RepoIntelligencePipeline()

    def run(self):
        """Run interactive CLI."""
        print("=" * 60)
        print("🧠 GitHub Repository Intelligence System")
        print("=" * 60)

        while True:
            print("\nCommands:")
            print("  1. index <github_url>  - Index a repository")
            print("  2. ask <repo_name>     - Ask a question")
            print("  3. explain <repo_name> <function/file>")
            print("  4. bugs <repo_name>    - Find bugs")
            print("  5. arch <repo_name>    - Generate architecture")
            print("  6. stats <repo_name>   - View stats")
            print("  7. delete <repo_name>  - Remove from index")
            print("  8. quit                - Exit")

            command = input("\n> ").strip().split()
            if not command:
                continue

            try:
                if command[0] == "index" and len(command) > 1:
                    result = self.pipeline.index_repository(command[1])
                    print(f"\n✅ Indexed successfully!")
                    print(f"   Repository: {result['repo_name']}")
                    print(f"   Files: {result['total_files']}")
                    print(f"   Chunks: {result['total_chunks']}")

                elif command[0] == "ask" and len(command) > 2:
                    repo_name = command[1]
                    question = " ".join(command[2:])
                    answer = self.pipeline.ask(repo_name, question)
                    print(f"\n🤖 {answer}")

                elif command[0] == "explain" and len(command) > 2:
                    repo_name = command[1]
                    target = command[2]
                    if "." in target:
                        result = self.pipeline.explain_file(repo_name, target)
                    else:
                        result = self.pipeline.explain_function(repo_name, target)
                    print(f"\n📖 {result}")

                elif command[0] == "bugs" and len(command) > 1:
                    bugs = self.pipeline.find_bugs(command[1])
                    print(f"\n🐛 Found {len(bugs)} potential issues:")
                    for bug in bugs:
                        print(f"   - {bug.get('title', 'Unknown')}")

                elif command[0] == "arch" and len(command) > 1:
                    result = self.pipeline.generate_architecture(command[1])
                    print(f"\n🏗️ Architecture Analysis Generated!")
                    print(f"   Component Diagram: {result['diagrams']['component']}")
                    print(f"   Class Diagram: {result['diagrams']['class']}")
                    print(f"   Mermaid: Available in output")

                elif command[0] == "stats" and len(command) > 1:
                    stats = self.pipeline.get_repo_stats(command[1])
                    print(f"\n📊 Stats: {json.dumps(stats, indent=2)}")

                elif command[0] == "delete" and len(command) > 1:
                    self.pipeline.delete_repo(command[1])
                    print(f"\n🗑️ Deleted {command[1]} from index")

                elif command[0] == "quit":
                    print("👋 Goodbye!")
                    break

                else:
                    print("❌ Invalid command")

            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    cli = CLI()
    cli.run()
