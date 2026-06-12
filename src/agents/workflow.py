"""
LangGraph Agent Workflow System
Implements graph-based reasoning for code analysis tasks.
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
import json

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from config.settings import get_settings
from src.vector_db.store import VectorStore
from src.embeddings.embedder import CodeEmbedder

# State Definition
class AgentState(TypedDict):
    messages: Annotated[List[Any], add]
    query: str
    context: List[Dict]
    analysis: str
    task_type: str  # "question", "architecture", "explain", "bug_find"
    repo_name: str
    iteration_count: int
    final_answer: str

class RepoIntelligenceAgent:
    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.model_name,
            temperature=self.settings.temperature,
            google_api_key=self.settings.gemini_api_key
        )
        self.vector_store = VectorStore()
        self.embedder = CodeEmbedder()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("classify", self._classify_task)
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("analyze", self._analyze_code)
        workflow.add_node("reason", self._graph_reasoning)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("verify", self._verify_response)

        # Define edges
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_conditional_edges(
            "analyze",
            self._should_reason,
            {
                "reason": "reason",
                "generate": "generate"
            }
        )
        workflow.add_edge("reason", "generate")
        workflow.add_edge("generate", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._should_continue,
            {
                "continue": "retrieve",
                "end": END
            }
        )

        return workflow.compile()

    def _classify_task(self, state: AgentState) -> AgentState:
        """Classify the user query into a task type."""
        query = state["query"].lower()

        if any(word in query for word in ["architecture", "structure", "diagram", "overview", "how is it organized"]):
            task_type = "architecture"
        elif any(word in query for word in ["explain", "what does", "how does", "describe"]):
            task_type = "explain"
        elif any(word in query for word in ["bug", "error", "issue", "fix", "wrong", "problem"]):
            task_type = "bug_find"
        else:
            task_type = "question"

        state["task_type"] = task_type
        state["messages"].append(AIMessage(content=f"Task classified as: {task_type}"))
        return state

    def _retrieve_context(self, state: AgentState) -> AgentState:
        """Retrieve relevant code context from vector DB."""
        query = state["query"]
        repo_name = state["repo_name"]

        # Embed query
        query_embedding = self.embedder.embed_query(query).tolist()

        # Search with filters for repo
        results = self.vector_store.hybrid_search(
            query_vector=query_embedding,
            query_text=query,
            top_k=10,
            filters={"repo_name": repo_name}
        )

        # Deduplicate and enrich context
        seen_files = set()
        context = []
        for r in results:
            file_key = r["file_path"]
            if file_key not in seen_files:
                seen_files.add(file_key)
                context.append(r)

        state["context"] = context
        state["messages"].append(AIMessage(content=f"Retrieved {len(context)} relevant code chunks"))
        return state

    def _analyze_code(self, state: AgentState) -> AgentState:
        """Analyze retrieved code with LLM."""
        context = state["context"]
        query = state["query"]
        task_type = state["task_type"]

        # Build context string
        context_str = self._format_context(context)

        # Task-specific prompts
        prompts = {
            "architecture": """Analyze the code architecture. Identify:
- Main components and their relationships
- Design patterns used
- Data flow between modules
- Entry points and core abstractions

Code Context:
{context}

Provide a structured analysis.""",

            "explain": """Explain the following code in detail:
- What it does
- How it works step by step
- Key functions and their roles
- Dependencies and imports

Code Context:
{context}

User Question: {query}

Provide a clear explanation.""",

            "bug_find": """Analyze the code for potential bugs or issues:
- Logic errors
- Edge cases not handled
- Performance issues
- Security vulnerabilities
- Common anti-patterns

Code Context:
{context}

User Concern: {query}

List specific issues with line references.""",

            "question": """Answer the question based on the code context:

Code Context:
{context}

Question: {query}

Provide a detailed, accurate answer with code references."""
        }

        prompt = prompts.get(task_type, prompts["question"]).format(
            context=context_str,
            query=query
        )

        messages = [
            SystemMessage(content="You are an expert code analyst. Be precise and reference specific code sections."),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        state["analysis"] = response.content
        state["messages"].append(AIMessage(content="Analysis complete"))
        return state

    def _graph_reasoning(self, state: AgentState) -> AgentState:
        """Perform graph-based reasoning for complex queries."""
        # Build a dependency graph from context
        context = state["context"]

        # Extract relationships
        nodes = {}
        edges = []

        for item in context:
            file_path = item["file_path"]
            if file_path not in nodes:
                nodes[file_path] = {
                    "type": "file",
                    "chunks": [],
                    "imports": item.get("context", {}).get("imports", [])
                }
            nodes[file_path]["chunks"].append(item)

            # Find cross-file references
            content = item["content"]
            for other_file in nodes:
                if other_file != file_path:
                    other_name = other_file.split("/")[-1].replace(".py", "").replace(".js", "")
                    if other_name in content:
                        edges.append((file_path, other_file, "references"))

        # Use LLM to reason over the graph
        graph_desc = f"""Dependency Graph:
Nodes (Files): {list(nodes.keys())}
Edges (Relationships): {edges}

Analyze the relationships and dependencies. Identify:
- Central/hub files
- Circular dependencies
- Isolation levels
- Critical paths
"""

        messages = [
            SystemMessage(content="You are a software architect. Analyze code dependencies and relationships."),
            HumanMessage(content=graph_desc + "\n\n" + state["analysis"])
        ]

        response = self.llm.invoke(messages)
        state["analysis"] = response.content
        state["messages"].append(AIMessage(content="Graph reasoning complete"))
        return state

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response based on analysis."""
        task_type = state["task_type"]
        analysis = state["analysis"]
        context = state["context"]

        # Format references
        references = []
        for item in context[:5]:
            ref = f"- `{item['file_path']}` (lines {item['start_line']}-{item['end_line']})"
            references.append(ref)

        if task_type == "architecture":
            final = f"""## Architecture Analysis

{analysis}

### Key Files
{chr(10).join(references)}
"""
        elif task_type == "bug_find":
            final = f"""## Bug Analysis

{analysis}

### References
{chr(10).join(references)}
"""
        else:
            final = f"""## Analysis

{analysis}

### Source References
{chr(10).join(references)}
"""

        state["final_answer"] = final
        state["messages"].append(AIMessage(content="Response generated"))
        return state

    def _verify_response(self, state: AgentState) -> AgentState:
        """Verify response quality and determine if more context needed."""
        state["iteration_count"] = state.get("iteration_count", 0) + 1

        # Simple verification: check if answer is too short or mentions needing more info
        answer = state["final_answer"]
        if len(answer) < 200 and state["iteration_count"] < 3:
            state["messages"].append(AIMessage(content="Need more context"))
        else:
            state["messages"].append(AIMessage(content="Verification passed"))

        return state

    def _should_reason(self, state: AgentState) -> str:
        """Determine if graph reasoning is needed."""
        if state["task_type"] == "architecture" and len(state["context"]) > 3:
            return "reason"
        return "generate"

    def _should_continue(self, state: AgentState) -> str:
        """Determine if workflow should continue or end."""
        if state["iteration_count"] >= self.settings.max_iterations:
            return "end"

        last_message = state["messages"][-1].content if state["messages"] else ""
        if "Need more context" in last_message:
            return "continue"

        return "end"

    def _format_context(self, context: List[Dict]) -> str:
        """Format context for LLM prompt."""
        parts = []
        for item in context:
            part = f"""File: {item['file_path']} (lines {item['start_line']}-{item['end_line']})
Type: {item['chunk_type']}
```
{item['content']}
```
"""
            parts.append(part)
        return "\n---\n".join(parts)

    def ask(self, query: str, repo_name: str) -> str:
        """Main entry point to ask questions about a repository."""
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            query=query,
            context=[],
            analysis="",
            task_type="",
            repo_name=repo_name,
            iteration_count=0,
            final_answer=""
        )

        result = self.workflow.invoke(initial_state)
        return result["final_answer"]


# Specialized Agents
class ArchitectureAgent:
    """Agent specialized in generating architecture analysis."""

    def __init__(self, repo_intelligence: RepoIntelligenceAgent):
        self.agent = repo_intelligence

    def analyze(self, repo_name: str) -> Dict:
        """Generate comprehensive architecture analysis."""
        # Query for different architectural aspects
        aspects = [
            "What are the main entry points and core modules?",
            "How do the components communicate with each other?",
            "What design patterns are used in this codebase?",
            "What is the data flow architecture?"
        ]

        analyses = {}
        for aspect in aspects:
            analyses[aspect] = self.agent.ask(aspect, repo_name)

        return {
            "overview": analyses[aspects[0]],
            "communication": analyses[aspects[1]],
            "patterns": analyses[aspects[2]],
            "data_flow": analyses[aspects[3]]
        }

class BugFinderAgent:
    """Agent specialized in finding bugs and issues."""

    def __init__(self, repo_intelligence: RepoIntelligenceAgent):
        self.agent = repo_intelligence

    def find_bugs(self, repo_name: str, specific_file: Optional[str] = None) -> List[Dict]:
        """Find potential bugs in the codebase."""
        query = "Find potential bugs, errors, and issues in the code"
        if specific_file:
            query += f" in {specific_file}"

        result = self.agent.ask(query, repo_name)

        # Parse structured bug reports
        bugs = []
        # Simple parsing - in production, use structured output
        lines = result.split("\n")
        current_bug = {}

        for line in lines:
            if line.startswith("##") or line.startswith("###"):
                if current_bug:
                    bugs.append(current_bug)
                current_bug = {"title": line.strip("# "), "details": []}
            elif line.strip() and current_bug:
                current_bug["details"].append(line.strip())

        if current_bug:
            bugs.append(current_bug)

        return bugs
