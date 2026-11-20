"""
DocGenius: Advanced AI-Powered PDF Document Assistant & RAG Engine.
Features:
- Multi-document page-by-page extraction & metadata preservation
- Contextual heading fusion & boundary-aware chunking
- Neighbor-chunk retrieval context expansion (window stitching)
- Normalized FAISS Cosine Similarity retrieval
- Strict answer grounding with verifiable source citations
- Document Tools: Semantic Search, Hierarchical Summarization, Key-Point Extraction, Multi-Doc Comparison
- Real RAG Evaluation Suite (Precision@K, Recall@K, MRR, Grounding Score)
- Dedicated Developer / Retrieval Diagnostics Inspector
- Enterprise-grade, modern responsive Streamlit UI
"""

import os
import sys
import io
import time
from typing import List, Dict, Any, Optional
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.pdf_processor import process_pdf, process_multiple_pdfs, ProcessedDocument
from core.chunking import chunk_document, chunk_multiple_documents, DocumentChunk
from core.rag_engine import (
    RAGEngine,
    SearchResult,
    DEFAULT_GEN_MODEL,
    FALLBACK_GEN_MODELS,
    DEFAULT_EMBED_MODEL,
    STRICT_REFUSAL_PHRASE
)
from core.tools import DocumentTools
from core.evaluation import (
    TestCase,
    RAGEvaluator,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_context_grounding_score,
    compute_answer_relevance_score
)

# Load environment
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
icon_path = os.path.join(PROJECT_ROOT, "Artifacts", "Image Resources", "images.jpeg")
page_icon = "📄"
if os.path.exists(icon_path):
    try:
        page_icon = Image.open(icon_path)
    except Exception:
        page_icon = "📄"

st.set_page_config(
    page_title="DocGenius | AI-Powered Document Assistant",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS (Preserving brand colors #071c28 & #00e5ff)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main Background & Typography */
    .stApp {
        background-color: #071c28;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Brand Header */
    .brand-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .brand-accent {
        color: #00e5ff;
    }

    .brand-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Cards & Containers */
    .metric-card {
        background-color: #102b3a;
        border: 1px solid #1e3f52;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00e5ff;
        margin-top: 0.2rem;
    }

    /* Source Citation Card */
    .citation-card {
        background-color: #0c2330;
        border-left: 3px solid #00e5ff;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.6rem 0;
        font-size: 0.92rem;
    }

    .citation-meta {
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 0.3rem;
        display: flex;
        justify-content: space-between;
    }

    .citation-snippet {
        color: #cbd5e1;
        background-color: #081720;
        padding: 0.6rem;
        border-radius: 4px;
        border: 1px solid #1a3646;
        line-height: 1.45;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #13394d;
        color: #38bdf8;
        border: 1px solid #0284c7;
    }

    .badge-success {
        background-color: #064e3b;
        color: #34d399;
        border-color: #059669;
    }

    .badge-warning {
        background-color: #78350f;
        color: #fcd34d;
        border-color: #d97706;
    }

    .badge-info {
        background-color: #1e1b4b;
        color: #a5b4fc;
        border-color: #4338ca;
    }

    /* File Uploader Customization */
    [data-testid="stFileUploader"] {
        background-color: #102b3a;
        border: 2px dashed #00e5ff;
        border-radius: 14px;
        padding: 16px;
    }

    /* Inputs */
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
        background-color: #102b3a !important;
        color: #ffffff !important;
        border: 1px solid #294655 !important;
        border-radius: 8px !important;
    }

    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
        border-color: #00e5ff !important;
        box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.2) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1e3f52;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #102b3a !important;
        color: #00e5ff !important;
        border-bottom: 2px solid #00e5ff !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #071c28;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e3f52;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00e5ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eval_history" not in st.session_state:
    st.session_state.eval_history = []
if "last_processed_file_hashes" not in st.session_state:
    st.session_state.last_processed_file_hashes = []

# ---------------------------------------------------------
# Sidebar Controls & Pipeline Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")

    api_key_env = os.getenv("GEMINI_API_KEY")
    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=api_key_env if api_key_env else "",
        type="password",
        help="Reads from .env by default. Enter a key here to override."
    )

    if user_api_key:
        st.session_state.rag_engine.api_key = user_api_key
        import google.generativeai as genai
        genai.configure(api_key=user_api_key)

    st.divider()

    st.markdown("#### 🧠 Model Selection")
    gen_model = st.selectbox(
        "Generative LLM",
        options=FALLBACK_GEN_MODELS,
        index=0,
        help="Primary model used for grounded QA, summarization, and tools."
    )
    st.session_state.rag_engine.gen_model = gen_model

    st.markdown("#### ✂️ Chunking Strategy")
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=300,
        max_value=1500,
        value=800,
        step=50,
        help="Maximum characters per chunk. Boundary-aware split protects sentences."
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=300,
        value=150,
        step=25,
        help="Characters preserved across consecutive chunks for context continuity."
    )

    st.markdown("#### 🔍 Retrieval Tuning")
    top_k = st.slider(
        "Top-K Chunks",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        help="Number of most similar chunks to pass to the generator."
    )
    similarity_threshold = st.slider(
        "Min Similarity Threshold",
        min_value=0.10,
        max_value=0.70,
        value=0.25,
        step=0.05,
        help="Cosine similarity cutoff to filter out irrelevant chunks."
    )

    expand_context_setting = st.checkbox(
        "Context Expansion (Neighbor Stitching)",
        value=True,
        help="Automatically retrieves neighboring chunks from the same document to ensure headings and explanations remain united."
    )

    st.divider()

    st.markdown("#### 🔬 Developer Mode")
    dev_mode = st.checkbox(
        "Enable Diagnostics Inspector",
        value=False,
        help="Enables low-level inspection of page text, chunks, vector scores, and retrieval thresholds."
    )

    st.divider()

    st.markdown("#### 📊 Vector Index Status")
    is_indexed = st.session_state.rag_engine.is_indexed()
    if is_indexed:
        st.markdown(f'<span class="badge badge-success">✓ Ready: {len(st.session_state.chunks)} Chunks</span>', unsafe_allow_html=True)
        st.markdown(f"**Index**: FAISS IndexFlatIP (Cosine)")
        st.markdown(f"**Embed Dim**: {st.session_state.rag_engine.embedding_dimension or 3072}")
    else:
        st.markdown('<span class="badge badge-warning">⚠ No Documents Indexed</span>', unsafe_allow_html=True)

    if st.button("🧹 Clear Session & Cache", use_container_width=True):
        st.session_state.processed_docs = []
        st.session_state.chunks = []
        st.session_state.rag_engine.clear()
        st.session_state.chat_history = []
        st.session_state.eval_history = []
        st.session_state.last_processed_file_hashes = []
        st.success("Session reset.")
        st.rerun()

    st.divider()
    st.caption("DocGenius v2.1 • Resilient Multi-Doc RAG")


# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="brand-title">
        <span>DocGenius</span><span class="brand-accent">.AI</span>
    </div>
    <div class="brand-subtitle">
        Enterprise-Grade Multi-Document RAG with Verifiable Source Citations & Analytical Tools.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Document Upload & Processing Section
# ---------------------------------------------------------
with st.expander("📂 Document Ingestion & Multi-PDF Upload", expanded=(not is_indexed)):
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload standard PDFs. Multi-page and multi-document processing supported."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        process_clicked = st.button("⚡ Process & Index", type="primary", use_container_width=True)

    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file before processing.")
        elif not st.session_state.rag_engine.is_configured():
            st.error("Gemini API key is required. Please set it in the sidebar or in your .env file.")
        else:
            with st.spinner("Extracting text and preserving metadata across all pages..."):
                docs = []
                for file_obj in uploaded_files:
                    doc = process_pdf(file_obj, doc_name=file_obj.name)
                    docs.append(doc)

                st.session_state.processed_docs = docs

                # Check for errors or extraction notes
                for d in docs:
                    if d.error_message:
                        st.warning(f"⚠️ {d.doc_name}: {d.error_message}")
                    for note in d.extraction_notes:
                        st.info(f"ℹ️ {d.doc_name}: {note}")

                # Contextual Chunking
                chunks = chunk_multiple_documents(
                    docs=docs,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    deduplicate=True
                )
                st.session_state.chunks = chunks

            if not chunks:
                st.error("No extractable chunks found. Check if the PDF is scanned or image-only.")
            else:
                progress_bar = st.progress(0, text="Generating Gemini embeddings and building FAISS index...")

                def update_progress(curr, total):
                    pct = int((curr / total) * 100)
                    progress_bar.progress(pct, text=f"Generating embeddings: {curr}/{total} chunks...")

                try:
                    count = st.session_state.rag_engine.build_index(
                        chunks=chunks,
                        progress_callback=update_progress
                    )
                    progress_bar.empty()
                    st.success(
                        f"Successfully processed {len(docs)} document(s) into {len(chunks)} chunks "
                        f"and indexed in FAISS!"
                    )
                    time.sleep(0.5)
                    st.rerun()
                except Exception as idx_err:
                    progress_bar.empty()
                    st.error(f"Failed to generate embeddings: {str(idx_err)}")

# ---------------------------------------------------------
# Document Intelligence Tools Instance
# ---------------------------------------------------------
doc_tools = DocumentTools(st.session_state.rag_engine, gen_model=gen_model)

# ---------------------------------------------------------
# Main Application Tabs
# ---------------------------------------------------------
tab_names = [
    "💬 Chat & Citations",
    "📊 Document Overview",
    "🛠️ Document Tools",
    "📈 RAG Evaluation",
    "🏛️ System Architecture"
]
if dev_mode:
    tab_names.append("🔬 Retrieval Diagnostics")

tabs = st.tabs(tab_names)
tab_chat = tabs[0]
tab_overview = tabs[1]
tab_tools = tabs[2]
tab_eval = tabs[3]
tab_arch = tabs[4]
tab_diag = tabs[5] if dev_mode else None

# =========================================================
# TAB 1: Chat & Citations
# =========================================================
with tab_chat:
    st.markdown("#### Grounded Conversational Q&A")
    st.caption("Answers are derived strictly from retrieved document chunks. Citations link directly to document names and page numbers.")

    if not st.session_state.rag_engine.is_indexed():
        st.info("👋 Upload and process your PDF documents above to begin questioning!")
    else:
        # Display existing conversation history
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            sources = msg.get("sources", [])
            is_grounded = msg.get("is_grounded", True)

            with st.chat_message(role):
                st.markdown(content)

                if role == "assistant" and sources:
                    with st.expander(f"📚 View {len(sources)} Verified Sources & Page Citations"):
                        for s in sources:
                            expanded_badge = '<span class="badge badge-info">Context Expanded</span>' if s.get("is_expanded") else ''
                            pages_display = s.get("pages_str") or str(s.get("page_number", 1))
                            snippet_text = s.get("expanded_snippet") or s.get("snippet", "")

                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <div class="citation-meta">
                                        <span>📄 {s['doc_name']} &nbsp;|&nbsp; Page(s): {pages_display} &nbsp; {expanded_badge}</span>
                                        <span class="badge">Relevance: {s.get('similarity_score', 0):.2f}</span>
                                    </div>
                                    <div class="citation-snippet">"{snippet_text}"</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

        # Chat Input
        query = st.chat_input("Ask a question about the uploaded document(s)...")

        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            available_names = [d.doc_name for d in st.session_state.processed_docs]
            intent = doc_tools.route_intent(query, available_names)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving verified document chunks..."):
                    retrieved = st.session_state.rag_engine.retrieve(
                        query=query,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        expand_context=expand_context_setting
                    )

                    ans_data = st.session_state.rag_engine.generate_grounded_answer(
                        query=query,
                        retrieved_results=retrieved,
                        model_name=gen_model
                    )

                    answer_text = ans_data["answer"]
                    sources = ans_data["sources"]
                    is_grounded = ans_data["is_grounded"]

                    st.markdown(answer_text)

                    if sources:
                        with st.expander(f"📚 View {len(sources)} Verified Sources & Page Citations"):
                            for s in sources:
                                expanded_badge = '<span class="badge badge-info">Context Expanded</span>' if s.get("is_expanded") else ''
                                pages_display = s.get("pages_str") or str(s.get("page_number", 1))
                                snippet_text = s.get("expanded_snippet") or s.get("snippet", "")

                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <div class="citation-meta">
                                            <span>📄 {s['doc_name']} &nbsp;|&nbsp; Page(s): {pages_display} &nbsp; {expanded_badge}</span>
                                            <span class="badge">Relevance: {s.get('similarity_score', 0):.2f}</span>
                                        </div>
                                        <div class="citation-snippet">"{snippet_text}"</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "is_grounded": is_grounded
                    })

# =========================================================
# TAB 2: Document Overview & Stats
# =========================================================
with tab_overview:
    st.markdown("#### Document Intelligence & Corpus Statistics")

    if not st.session_state.processed_docs:
        st.info("No documents uploaded yet.")
    else:
        total_docs = len(st.session_state.processed_docs)
        total_pages = sum(d.total_pages for d in st.session_state.processed_docs)
        total_words = sum(d.total_words for d in st.session_state.processed_docs)
        total_chunks = len(st.session_state.chunks)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Documents</div>
                    <div class="metric-value">{total_docs}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Pages</div>
                    <div class="metric-value">{total_pages}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Words</div>
                    <div class="metric-value">{total_words:,}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Vector Chunks</div>
                    <div class="metric-value">{total_chunks}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # Per document breakdown
        for doc in st.session_state.processed_docs:
            with st.expander(f"📄 {doc.doc_name} — Details & Page-by-Page Content"):
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Pages", f"{doc.non_empty_pages} / {doc.total_pages}")
                col_m2.metric("Characters", f"{doc.total_chars:,}")
                col_m3.metric("Words", f"{doc.total_words:,}")
                col_m4.metric("Est. Read Time", f"{doc.estimated_read_time_mins} min")

                if doc.error_message:
                    st.warning(f"⚠️ {doc.error_message}")
                for note in doc.extraction_notes:
                    st.info(f"ℹ️ {note}")

                st.markdown("##### Extracted Pages:")
                for p in doc.pages:
                    density_tag = " [Low Density Snippet]" if p.is_low_density else ""
                    status = "⚪ Empty" if p.is_empty else f"🟢 {p.word_count} words ({p.char_count} chars){density_tag}"
                    with st.expander(f"Page {p.page_number} ({status})"):
                        if p.is_empty:
                            st.caption("No selectable text extracted from this page.")
                        else:
                            st.text_area(
                                label=f"Page {p.page_number} Text",
                                value=p.text,
                                height=120,
                                disabled=True,
                                key=f"preview_{doc.doc_name}_{p.page_number}"
                            )

# =========================================================
# TAB 3: Document Tools
# =========================================================
with tab_tools:
    st.markdown("#### Document Analytics & Processing Tools")
    st.caption("Directly invoke analytical tools for deep document understanding.")

    if not st.session_state.processed_docs:
        st.info("Upload and process documents to enable analytical tools.")
    else:
        doc_names = [d.doc_name for d in st.session_state.processed_docs]

        tool_choice = st.radio(
            "Select Tool",
            [
                "🔎 Semantic Vector Search",
                "📝 Summarize Document",
                "💡 Extract Key Points",
                "⚖️ Compare Documents"
            ],
            horizontal=True
        )

        st.divider()

        if tool_choice == "🔎 Semantic Vector Search":
            st.markdown("##### Direct Semantic Vector Search")
            st.caption("Search directly through FAISS vector space to inspect chunk rankings and similarity scores.")

            search_col1, search_col2 = st.columns([3, 1])
            with search_col1:
                search_query = st.text_input("Search query:")
            with search_col2:
                search_k = st.number_input("Top-K results", min_value=1, max_value=10, value=5)

            if st.button("Run Vector Search", type="primary"):
                if not search_query.strip():
                    st.warning("Please enter a query.")
                else:
                    with st.spinner("Searching vector index..."):
                        search_res = doc_tools.search_document(search_query, top_k=search_k)

                        if search_res["status"] == "success":
                            st.success(f"Found {search_res['total_matches']} relevant matches:")
                            for item in search_res["results"]:
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <div class="citation-meta">
                                            <span>#{item['rank']} • 📄 {item['document']} (Page {item['page']})</span>
                                            <span class="badge">Similarity: {item['similarity_score']:.4f}</span>
                                        </div>
                                        <div class="citation-snippet">{item['snippet']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.error(search_res["message"])

        elif tool_choice == "📝 Summarize Document":
            st.markdown("##### Document Summarization (Hierarchical Map-Reduce)")
            selected_doc = st.selectbox("Select document to summarize:", doc_names)
            sum_type = st.selectbox(
                "Summary Style:",
                ["Executive Summary", "Comprehensive Technical Summary", "Action-Oriented Brief"]
            )

            if st.button("Generate Summary", type="primary"):
                with st.spinner("Analyzing document structure and synthesizing summary..."):
                    res = doc_tools.summarize_document(selected_doc, st.session_state.processed_docs, summary_type=sum_type)
                    if res["status"] == "success":
                        st.markdown(f"**Method**: `{res.get('method', 'Direct')}` | **Pages Analyzed**: `{res.get('pages_analyzed', 0)}`")
                        st.markdown(res["summary"])
                    else:
                        st.error(res["message"])

        elif tool_choice == "💡 Extract Key Points":
            st.markdown("##### Extract Key Findings, Metrics & Conclusions")
            selected_doc = st.selectbox("Select document for key-point extraction:", doc_names)

            if st.button("Extract Key Points", type="primary"):
                with st.spinner("Extracting critical facts, data points, and conclusions..."):
                    res = doc_tools.extract_key_points(selected_doc, st.session_state.processed_docs)
                    if res["status"] == "success":
                        st.markdown(res["key_points"])
                    else:
                        st.error(res["message"])

        elif tool_choice == "⚖️ Compare Documents":
            st.markdown("##### Multi-Document Comparative Analysis")
            if len(doc_names) < 2:
                st.warning("Document comparison requires at least 2 uploaded PDFs.")
            else:
                selected_compare_docs = st.multiselect(
                    "Select documents to compare:",
                    doc_names,
                    default=doc_names[:2]
                )
                focus_area = st.text_input(
                    "Specific focus or comparison theme (optional):",
                    placeholder="e.g. Methodology, financial results, conclusions"
                )

                if st.button("Compare Documents", type="primary"):
                    if len(selected_compare_docs) < 2:
                        st.warning("Please select at least 2 documents.")
                    else:
                        with st.spinner("Synthesizing multi-document comparison matrix..."):
                            res = doc_tools.compare_documents(
                                selected_compare_docs,
                                st.session_state.processed_docs,
                                comparison_focus=focus_area
                            )
                            if res["status"] == "success":
                                st.markdown(res["comparison"])
                            else:
                                st.error(res["message"])

# =========================================================
# TAB 4: RAG Evaluation Dashboard
# =========================================================
with tab_eval:
    st.markdown("#### RAG Evaluation & Quality Benchmarking")
    st.caption("Measure retrieval precision, keyword recall, mean reciprocal rank, and context faithfulness with mathematically computed metrics.")

    if not st.session_state.rag_engine.is_indexed():
        st.info("Upload and index documents to execute RAG evaluations.")
    else:
        evaluator = RAGEvaluator(st.session_state.rag_engine)

        eval_mode = st.radio(
            "Evaluation Method",
            ["Interactive Benchmark Test", "Custom Single-Query Metric Evaluation"],
            horizontal=True
        )

        st.divider()

        if eval_mode == "Interactive Benchmark Test":
            st.markdown("##### Pre-configured Document Verification Suite")
            first_doc = st.session_state.processed_docs[0]
            sample_words = [w for w in first_doc.get_full_text().split() if len(w) > 4][:5]
            sample_term = sample_words[0] if sample_words else "document"

            default_q = f"What does the document state about {sample_term}?"

            test_q = st.text_input("Benchmark Question:", value=default_q)
            target_keywords_input = st.text_input(
                "Target Keywords for Retrieval Relevance (comma-separated):",
                value=f"{sample_term}, result, analysis"
            )
            expected_ans = st.text_area(
                "Expected Reference Answer (optional):",
                value="The document discusses relevant findings and methodologies."
            )

            if st.button("Run Benchmark Evaluation", type="primary"):
                k_eval = top_k
                kws = [k.strip() for k in target_keywords_input.split(",") if k.strip()]

                tc = TestCase(
                    question=test_q,
                    expected_answer=expected_ans,
                    target_keywords=kws
                )

                with st.spinner("Executing retrieval, generation, and metric evaluation..."):
                    result = evaluator.evaluate_test_case(tc, top_k=k_eval)

                    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                    col_e1.metric(f"Precision@{k_eval}", f"{result.precision_at_k * 100:.1f}%")
                    col_e2.metric(f"Recall@{k_eval}", f"{result.recall_at_k * 100:.1f}%")
                    col_e3.metric("Reciprocal Rank (MRR)", f"{result.reciprocal_rank:.2f}")
                    col_e4.metric("Grounding Score", f"{result.grounding_score * 100:.1f}%")

                    st.markdown("---")
                    st.markdown("##### Detailed Evaluation Report")
                    st.markdown(f"**Test Question**: {result.question}")
                    st.markdown(f"**Generated Answer**: {result.generated_answer}")
                    st.markdown(f"**Grounding Verification**: {'✅ Fully Grounded' if result.is_grounded else '❌ Refusal / Ungrounded'}")
                    st.markdown(f"**Evaluator Notes**: `{result.notes}`")

                    st.markdown("##### Retrieved Source Chunks Evaluated:")
                    for idx, s in enumerate(result.retrieved_sources):
                        st.markdown(
                            f"- **Chunk {idx+1}** (Doc: `{s['document']}`, Page: `{s['page']}`, Similarity: `{s['similarity_score']:.3f}`): {s['text_preview']}"
                        )

        else:
            st.markdown("##### Custom Query Grounding & Relevance Check")
            custom_q = st.text_input("Query to evaluate:")
            custom_context = st.text_area("Reference context or leave blank to use retrieved context:")

            if st.button("Evaluate Query Grounding", type="primary"):
                if not custom_q.strip():
                    st.warning("Enter a query.")
                else:
                    with st.spinner("Evaluating..."):
                        retrieved = st.session_state.rag_engine.retrieve(custom_q, top_k=top_k)
                        ans_data = st.session_state.rag_engine.generate_grounded_answer(custom_q, retrieved)

                        ctx = custom_context if custom_context.strip() else ans_data.get("context_used", "")
                        grounding = compute_context_grounding_score(ans_data["answer"], ctx)
                        relevance = compute_answer_relevance_score(ans_data["answer"], custom_q)

                        col1, col2 = st.columns(2)
                        col1.metric("Context Grounding Score", f"{grounding * 100:.1f}%")
                        col2.metric("Answer-Query Relevance", f"{relevance * 100:.1f}%")

                        st.markdown(f"**Answer**: {ans_data['answer']}")

        st.divider()
        with st.expander("ℹ️ Understanding RAG Evaluation Metrics & Limitations"):
            st.markdown(
                """
                - **Precision@K**: Measures the proportion of the top-K retrieved chunks that contain target relevant concepts.
                - **Recall@K**: Measures whether the complete set of required target concepts were captured across the top-K retrieved chunks.
                - **Reciprocal Rank (RR)**: Reflects how early the first relevant chunk appeared in the ranked results (1.0 = Rank 1, 0.5 = Rank 2).
                - **Context Grounding / Faithfulness**: Computes sentence-level token overlap against retrieved context to prevent hallucination.
                - **Evaluation Limitations**: Lexical overlap metrics provide an objective lower bound for factual presence without hallucination risk.
                """
            )

# =========================================================
# TAB 5: System Architecture & Technical Specifications
# =========================================================
with tab_arch:
    st.markdown("#### DocGenius System Architecture & RAG Pipeline")
    st.caption("Technical architectural breakdown of the upgraded pipeline.")

    st.markdown(
        """
        ```mermaid
        flowchart TD
            subgraph INGESTION ["1. Ingestion & Quality Analysis"]
                A[Multi-PDF Upload Stream] --> B[PyPDF2 Text Normalizer]
                B --> C[Page Metadata & Low-Density Tagging]
            end

            subgraph CHUNKING ["2. Contextual Chunking & Fusion"]
                C --> D[Heading & Fragment Fusion Engine]
                D --> E[Recursive Boundary-Aware Splitter]
                E --> F[MD5 Deduplication Filter]
            end

            subgraph VECTORSTORE ["3. Normalized Vector Index"]
                F --> G[Gemini 3072-dim Embeddings]
                G --> H[Unit L2 Normalization]
                H --> I[FAISS IndexFlatIP Cosine Space]
            end

            subgraph RETRIEVAL ["4. Retrieval & Context Expansion"]
                J[User Query] --> K[Query Embedding & Normalization]
                K --> L[FAISS Cosine Similarity Search]
                L --> M[Neighbor-Chunk Window Stitching]
                M --> N[Top-K Expanded Context Chunks]
            end

            subgraph GENERATION ["5. Resilient Generation"]
                N --> O[Context Assembler + Citations]
                O --> P[Gemini Resilient Model Fallback Chain]
                P --> Q[Grounded Answer + Page Provenance]
            end
        ```
        """,
        unsafe_allow_html=True
    )

# =========================================================
# TAB 6 (Optional): Developer / Retrieval Diagnostics Inspector
# =========================================================
if dev_mode and tab_diag:
    with tab_diag:
        st.markdown("#### 🔬 Developer Diagnostics & Retrieval Inspector")
        st.caption("Inspect raw page extractions, generated chunks, similarity scores, and neighbor expansion logic.")

        diag_section = st.radio(
            "Diagnostic Inspection Area",
            ["📑 Raw Page Extractions", "🧩 Generated Chunks & Metadata Registry", "🎯 Live Retrieval & Threshold Inspector"],
            horizontal=True
        )

        st.divider()

        if diag_section == "📑 Raw Page Extractions":
            st.markdown("##### Extracted Page Inspection")
            if not st.session_state.processed_docs:
                st.info("No documents uploaded.")
            else:
                for doc in st.session_state.processed_docs:
                    st.markdown(f"**Document**: `{doc.doc_name}` | Total Chars: `{doc.total_chars}` | Non-empty Pages: `{doc.non_empty_pages}`")
                    for p in doc.pages:
                        density_label = "⚠️ LOW DENSITY (<60 chars)" if p.is_low_density else "🟢 Standard"
                        with st.expander(f"Page {p.page_number} ({len(p.text)} chars, {p.word_count} words) — {density_label}"):
                            st.code(p.text if p.text else "[EMPTY STRING / NO SELECTABLE TEXT]")

        elif diag_section == "🧩 Generated Chunks & Metadata Registry":
            st.markdown("##### Chunk Registry & Fusion Inspection")
            if not st.session_state.chunks:
                st.info("No chunks indexed.")
            else:
                st.markdown(f"Total Chunks: `{len(st.session_state.chunks)}`")
                for c in st.session_state.chunks:
                    heading_tag = "📌 Has Heading Context" if c.has_heading_context else ""
                    pages_str = ", ".join(str(p) for p in (c.spanned_pages or [c.page_number]))
                    with st.expander(f"[{c.chunk_index}] ID: `{c.chunk_id}` | Page(s): {pages_str} | {c.char_count} chars {heading_tag}"):
                        st.markdown(f"**Text Content:**")
                        st.code(c.text)

        elif diag_section == "🎯 Live Retrieval & Threshold Inspector":
            st.markdown("##### Live Retrieval Diagnostic Probe")
            probe_query = st.text_input("Test Query for Diagnostic Probe:", value="Explain the types of environment mentioned in the document.")
            probe_expand = st.checkbox("Simulate with Context Window Expansion", value=True)

            if st.button("Run Diagnostic Probe", type="primary"):
                if not st.session_state.rag_engine.is_indexed():
                    st.warning("Index is empty.")
                else:
                    results = st.session_state.rag_engine.retrieve(
                        query=probe_query,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        expand_context=probe_expand
                    )
                    diag = st.session_state.rag_engine.last_diagnostics

                    st.markdown(f"**Threshold Applied**: `{diag.get('threshold')}`")
                    st.markdown(f"**Total Candidates Evaluated**: `{diag.get('total_candidates_examined')}`")
                    st.markdown(f"**Accepted Results (Passed Threshold)**: `{len(results)}`")

                    st.markdown("##### Candidate Rankings:")
                    for c_cand in diag.get("candidates", []):
                        status = "✅ ACCEPTED" if c_cand["passed_threshold"] else "❌ FILTERED (Below Threshold)"
                        st.markdown(
                            f"- **Score: `{c_cand['score']:.4f}`** | ID: `{c_cand['chunk_id']}` (Page {c_cand['page']}) -> {status}"
                        )
                        st.caption(f"Snippet: \"{c_cand['snippet']}\"")

                    st.markdown("---")
                    st.markdown("##### Retrieved Context Blocks Sent to LLM:")
                    for r in results:
                        st.markdown(f"**Rank {r.rank}** | ID: `{r.chunk.chunk_id}` | Similarity: `{r.similarity_score:.4f}`")
                        if r.is_expanded:
                            st.info("ℹ️ Context was expanded with adjacent chunks:")
                            st.code(r.expanded_text)
                        else:
                            st.code(r.chunk.text)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #64748b; font-size: 0.85rem;">DocGenius AI • Advanced Retrieval-Augmented Generation Architecture</div>',
    unsafe_allow_html=True
)
