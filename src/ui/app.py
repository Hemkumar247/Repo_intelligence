"""
Streamlit Web Interface
Interactive UI for the Repository Intelligence System.
"""
from __future__ import annotations

import os

import streamlit as st

from src.services.runtime import get_pipeline

st.set_page_config(
    page_title="GitHub Repo Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; }
    .sub-header { font-size: 1.1rem; color: #666; }
    .trust-note {
        border-left: 4px solid #1f77b4;
        background: #f7fbff;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_pipeline():
    return get_pipeline()


def render_answer_details(result: dict) -> None:
    confidence = result.get("confidence", "unknown").upper()
    st.markdown(f'<div class="trust-note"><strong>Confidence:</strong> {confidence}</div>', unsafe_allow_html=True)
    st.markdown(result["answer"])

    sources = result.get("sources", [])
    if sources:
        st.markdown("### Citations")
        for source in sources:
            score = source.get("score")
            score_text = f" • score {score:.3f}" if isinstance(score, (int, float)) else ""
            st.markdown(
                f"- `{source['file_path']}` lines {source['start_line']}-{source['end_line']}"
                f" • {source.get('chunk_type', 'code')}{score_text}"
            )

    previews = result.get("context_preview", [])
    if previews:
        with st.expander("Retrieved Context Preview"):
            for preview in previews:
                st.markdown(
                    f"**{preview['file_path']}** lines {preview['start_line']}-{preview['end_line']}"
                )
                st.code(preview["snippet"])


def repo_pipeline():
    return get_cached_pipeline()

with st.sidebar:
    st.title("Repo Intelligence")
    st.markdown("---")
    st.subheader("Repository")

    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo"
    )

    if st.button("Index Repository", type="primary"):
        if repo_url:
            with st.spinner("Indexing repository..."):
                try:
                    result = repo_pipeline().index_repository(repo_url)
                    st.session_state["current_repo"] = result["repo_name"]
                    st.session_state["index_result"] = result
                    st.success(
                        f"Indexed {result['repo_name']} with "
                        f"{result['total_files']} files and {result['total_chunks']} chunks."
                    )
                except Exception as exc:
                    st.error(f"Error: {exc}")
        else:
            st.warning("Please enter a repository URL")

    st.markdown("---")
    if "current_repo" in st.session_state:
        st.subheader("Current Repository")
        st.info(st.session_state["current_repo"])
        if st.button("Remove from Index"):
            repo_pipeline().delete_repo(st.session_state["current_repo"])
            del st.session_state["current_repo"]
            st.rerun()

st.markdown('<p class="main-header">GitHub Repository Intelligence</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Understand codebases with stronger citations, context previews, and trust signals.</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

if "current_repo" not in st.session_state:
    st.info("Start by indexing a GitHub repository from the sidebar.")
else:
    repo_name = st.session_state["current_repo"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Ask Questions",
        "Explain Code",
        "Architecture",
        "Find Bugs",
        "Statistics",
    ])

    with tab1:
        st.subheader("Ask Questions About the Code")
        question = st.text_area(
            "Your question",
            placeholder="How does authentication work?",
            height=100,
        )
        if st.button("Ask", type="primary"):
            if question:
                with st.spinner("Analyzing..."):
                    result = repo_pipeline().ask_with_sources(repo_name, question)
                    st.markdown("### Answer")
                    render_answer_details(result)
            else:
                st.warning("Please enter a question")

    with tab2:
        st.subheader("Explain Functions or Files")
        explain_type = st.radio("What to explain", ["Function", "File"], horizontal=True)
        if explain_type == "Function":
            function_name = st.text_input("Function name", placeholder="authenticate_user")
            if st.button("Explain Function"):
                if function_name:
                    with st.spinner("Analyzing function..."):
                        st.markdown(repo_pipeline().explain_function(repo_name, function_name))
                else:
                    st.warning("Enter a function name")
        else:
            file_path = st.text_input("File path", placeholder="src/auth.py")
            if st.button("Explain File"):
                if file_path:
                    with st.spinner("Analyzing file..."):
                        st.markdown(repo_pipeline().explain_file(repo_name, file_path))
                else:
                    st.warning("Enter a file path")

    with tab3:
        st.subheader("Architecture Analysis")
        if st.button("Generate Architecture Analysis", type="primary"):
            with st.spinner("Generating architecture analysis..."):
                result = repo_pipeline().generate_architecture(repo_name)
                st.markdown("### Overview")
                st.markdown(result["analysis"]["overview"])
                st.markdown("### Component Communication")
                st.markdown(result["analysis"]["communication"])
                st.markdown("### Design Patterns")
                st.markdown(result["analysis"]["patterns"])
                st.markdown("### Data Flow")
                st.markdown(result["analysis"]["data_flow"])
                st.markdown("### Diagrams")
                if os.path.exists(result["diagrams"]["component"]):
                    st.image(result["diagrams"]["component"], caption="Component Diagram")
                if os.path.exists(result["diagrams"]["class"]):
                    st.image(result["diagrams"]["class"], caption="Class Diagram")
                with st.expander("View Mermaid Diagram Code"):
                    st.code(result["diagrams"]["mermaid"], language="mermaid")

    with tab4:
        st.subheader("Bug Detection")
        specific_file = st.text_input("Specific file (optional)", placeholder="src/main.py")
        if st.button("Find Bugs", type="primary"):
            with st.spinner("Scanning for bugs..."):
                bugs = repo_pipeline().find_bugs(repo_name, specific_file or None)
                if bugs:
                    st.error(f"Found {len(bugs)} potential issues")
                    for index, bug in enumerate(bugs, start=1):
                        title = bug.get("title", "Unknown issue")
                        severity = bug.get("severity", "unknown")
                        location = f"{bug.get('file_path', 'unknown file')}:{bug.get('start_line', '?')}"
                        with st.expander(f"Issue #{index}: {title} [{severity}]"):
                            st.markdown(f"**Location:** `{location}`")
                            st.markdown(bug.get("description", "No description provided."))
                            if bug.get("recommendation"):
                                st.markdown(f"**Recommendation:** {bug['recommendation']}")
                else:
                    st.success("No obvious bugs detected. Manual review is still recommended.")

    with tab5:
        st.subheader("Repository Statistics")
        stats = repo_pipeline().get_repo_stats(repo_name)

        col1, col2, col3 = st.columns(3)
        col1.metric("Points", stats.get("points_count", 0))
        col2.metric("Files", stats.get("files_count", 0))
        col3.metric("Scope", stats.get("scope", "repository"))

        st.markdown("### Indexing Summary")
        st.json(stats)

        if "index_result" in st.session_state:
            st.markdown("### Last Index Result")
            st.json(st.session_state["index_result"])
