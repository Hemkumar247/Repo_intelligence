"""
Streamlit Web Interface
Interactive UI for the Repository Intelligence System.
"""
import streamlit as st
import os
import json
from pathlib import Path

import sys
from pathlib import Path

# Add project root to sys.path so 'src' package is discoverable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline import RepoIntelligencePipeline

# Page config
st.set_page_config(
    page_title="GitHub Repo Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
    }
    .code-block {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 15px;
        font-family: monospace;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return RepoIntelligencePipeline()

pipeline = get_pipeline()

# Sidebar
with st.sidebar:
    st.image("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", width=50)
    st.title("Repo Intelligence")

    st.markdown("---")
    st.subheader("🔧 Configuration")

    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo"
    )

    if st.button("🚀 Index Repository", type="primary"):
        if repo_url:
            with st.spinner("Indexing repository... This may take a few minutes"):
                try:
                    result = pipeline.index_repository(repo_url)
                    st.session_state["current_repo"] = result["repo_name"]
                    st.session_state["index_result"] = result
                    st.success(f"✅ Indexed {result['repo_name']}!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.warning("Please enter a repository URL")

    st.markdown("---")

    if "current_repo" in st.session_state:
        st.subheader("📁 Current Repository")
        st.info(st.session_state["current_repo"])

        if st.button("🗑️ Remove from Index"):
            pipeline.delete_repo(st.session_state["current_repo"])
            del st.session_state["current_repo"]
            st.rerun()

# Main content
st.markdown('<p class="main-header">🧠 GitHub Repository Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Understand any codebase with AI-powered analysis</p>', unsafe_allow_html=True)

st.markdown("---")

# Tabs
if "current_repo" in st.session_state:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Ask Questions", 
        "📖 Explain Code", 
        "🏗️ Architecture",
        "🐛 Find Bugs",
        "📊 Statistics"
    ])

    repo_name = st.session_state["current_repo"]

    # Tab 1: Ask Questions
    with tab1:
        st.subheader("Ask Questions About the Code")

        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_area(
                "Your question",
                placeholder="e.g., How does the authentication system work?",
                height=100
            )
        with col2:
            st.markdown("### Examples")
            examples = [
                "How does error handling work?",
                "What design patterns are used?",
                "Explain the database layer",
                "How is caching implemented?"
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex}"):
                    question = ex

        if st.button("🔍 Ask", type="primary"):
            if question:
                with st.spinner("Analyzing..."):
                    answer = pipeline.ask(repo_name, question)
                    st.markdown("### 🤖 Answer")
                    st.markdown(answer)
            else:
                st.warning("Please enter a question")

    # Tab 2: Explain Code
    with tab2:
        st.subheader("Explain Functions or Files")

        explain_type = st.radio("What to explain", ["Function", "File"], horizontal=True)

        if explain_type == "Function":
            function_name = st.text_input("Function name", placeholder="e.g., authenticate_user")
            if st.button("📖 Explain Function"):
                if function_name:
                    with st.spinner("Analyzing function..."):
                        result = pipeline.explain_function(repo_name, function_name)
                        st.markdown(result)
                else:
                    st.warning("Enter a function name")
        else:
            file_path = st.text_input("File path", placeholder="e.g., src/auth.py")
            if st.button("📖 Explain File"):
                if file_path:
                    with st.spinner("Analyzing file..."):
                        result = pipeline.explain_file(repo_name, file_path)
                        st.markdown(result)
                else:
                    st.warning("Enter a file path")

    # Tab 3: Architecture
    with tab3:
        st.subheader("Architecture Analysis")

        if st.button("🏗️ Generate Architecture Analysis", type="primary"):
            with st.spinner("Generating architecture analysis..."):
                result = pipeline.generate_architecture(repo_name)

                # Display analysis
                st.markdown("### Overview")
                st.markdown(result["analysis"]["overview"])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Component Communication")
                    st.markdown(result["analysis"]["communication"])
                with col2:
                    st.markdown("### Design Patterns")
                    st.markdown(result["analysis"]["patterns"])

                st.markdown("### Data Flow")
                st.markdown(result["analysis"]["data_flow"])

                # Diagrams
                st.markdown("### 📐 Diagrams")

                if os.path.exists(result["diagrams"]["component"]):
                    st.image(result["diagrams"]["component"], caption="Component Diagram")

                if os.path.exists(result["diagrams"]["class"]):
                    st.image(result["diagrams"]["class"], caption="Class Diagram")

                # Mermaid
                with st.expander("View Mermaid Diagram Code"):
                    st.code(result["diagrams"]["mermaid"], language="mermaid")

                # Metrics
                st.markdown("### 📊 Dependency Metrics")
                metrics = result["metrics"]

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Components", metrics["total_components"])
                col2.metric("Total Dependencies", metrics["total_dependencies"])
                col3.metric("Avg Degree", f"{metrics['average_degree']:.2f}")

                st.markdown("#### Most Central Components")
                for comp, centrality in metrics["central_components"][:5]:
                    st.progress(centrality, text=f"{comp} ({centrality:.3f})")

    # Tab 4: Find Bugs
    with tab4:
        st.subheader("Bug Detection")

        specific_file = st.text_input(
            "Specific file (optional)",
            placeholder="Leave empty to scan entire repo"
        )

        if st.button("🐛 Find Bugs", type="primary"):
            with st.spinner("Scanning for bugs..."):
                file_filter = specific_file if specific_file else None
                bugs = pipeline.find_bugs(repo_name, file_filter)

                if bugs:
                    st.error(f"Found {len(bugs)} potential issues")
                    for i, bug in enumerate(bugs):
                        with st.expander(f"Issue #{i+1}: {bug.get('title', 'Unknown')}"):
                            st.markdown("\n".join(bug.get("details", [])))
                else:
                    st.success("No obvious bugs detected! (Always review manually)")

    # Tab 5: Statistics
    with tab5:
        st.subheader("Repository Statistics")

        stats = pipeline.get_repo_stats(repo_name)

        col1, col2, col3 = st.columns(3)
        col1.metric("Vectors in Index", stats.get("vectors_count", 0))
        col2.metric("Indexed Vectors", stats.get("indexed_vectors_count", 0))
        col3.metric("Points", stats.get("points_count", 0))

        if "index_result" in st.session_state:
            result = st.session_state["index_result"]
            st.markdown("### Indexing Results")
            st.json(result)

else:
    # No repo selected
    st.info("👈 Start by indexing a GitHub repository from the sidebar")

    # Demo section
    st.markdown("---")
    st.subheader("✨ Features")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        ### 💬 Ask Questions
        Natural language queries about any part of the codebase
        """)

    with col2:
        st.markdown("""
        ### 📖 Explain Code
        Detailed explanations of functions, classes, and files
        """)

    with col3:
        st.markdown("""
        ### 🏗️ Architecture
        Generate diagrams and analyze system design
        """)

    with col4:
        st.markdown("""
        ### 🐛 Find Bugs
        AI-powered bug detection and code review
        """)

    st.markdown("---")
    st.subheader("🚀 How It Works")

    steps = [
        ("1. Connect", "Paste any GitHub repository URL"),
        ("2. Index", "AI indexes and embeds the entire codebase"),
        ("3. Query", "Ask questions in natural language"),
        ("4. Analyze", "Get insights, diagrams, and bug reports")
    ]

    for title, desc in steps:
        st.markdown(f"**{title}**: {desc}")
