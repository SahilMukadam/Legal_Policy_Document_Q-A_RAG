"""
Legal Document Q&A — Streamlit Frontend

Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os
import uuid
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.retrieval.store_provider import get_vector_store
from src.chains.rag_chain import RAGChain


# ---- Page Config ----
st.set_page_config(
    page_title="Legal Document Q&A",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS ----
st.markdown("""
<style>
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 1rem !important;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        min-width: 340px !important;
        max-width: 380px !important;
        transition: min-width 0.3s ease, max-width 0.3s ease !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* ===== COLLAPSED SIDEBAR — show icon strip ===== */
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 60px !important;
        max-width: 60px !important;
    }

    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 60px !important;
        overflow: hidden !important;
    }

    /* Hide text content when collapsed, show only the collapse button area */
    [data-testid="stSidebar"][aria-expanded="false"] .stMarkdown,
    [data-testid="stSidebar"][aria-expanded="false"] .stDivider,
    [data-testid="stSidebar"][aria-expanded="false"] .stSelectbox,
    [data-testid="stSidebar"][aria-expanded="false"] .stSlider,
    [data-testid="stSidebar"][aria-expanded="false"] .stCheckbox,
    [data-testid="stSidebar"][aria-expanded="false"] .stFileUploader,
    [data-testid="stSidebar"][aria-expanded="false"] .stTextInput,
    [data-testid="stSidebar"][aria-expanded="false"] .stExpander,
    [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"][aria-expanded="false"] .stToggle {
        display: none !important;
    }

    /* Keep buttons visible but compact when collapsed */
    [data-testid="stSidebar"][aria-expanded="false"] .stButton > button {
        padding: 8px !important;
        font-size: 0 !important;
        min-height: 40px !important;
        width: 44px !important;
        margin: 4px auto !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }

    /* ===== SOURCE BOXES ===== */
    .source-box {
        background: rgba(100, 140, 200, 0.1) !important;
        border-left: 3px solid #5b9bf5 !important;
        padding: 14px 16px !important;
        margin: 8px 0 !important;
        border-radius: 0 12px 12px 0 !important;
        color: inherit !important;
        backdrop-filter: blur(10px) !important;
    }

    .source-box strong { color: #5b9bf5 !important; }
    .source-box em { color: inherit !important; opacity: 0.85; }

    /* ===== DOCUMENT CARDS ===== */
    .doc-item {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        margin: 4px 0 !important;
        color: inherit !important;
    }

    .doc-item .doc-name { font-weight: 500 !important; font-size: 0.85em !important; color: inherit !important; }
    .doc-item .doc-meta { font-size: 0.7em !important; opacity: 0.5 !important; color: inherit !important; }

    /* ===== METRICS ===== */
    .metric-row { display: flex; gap: 10px; margin: 10px 0; }

    .metric-badge {
        background: rgba(91, 155, 245, 0.1) !important;
        border: 1px solid rgba(91, 155, 245, 0.2) !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        flex: 1 !important;
        text-align: center !important;
        color: inherit !important;
    }

    .metric-badge .metric-value { font-size: 1.4em !important; font-weight: 700 !important; color: #5b9bf5 !important; }
    .metric-badge .metric-label { font-size: 0.7em !important; opacity: 0.5 !important; color: inherit !important; }

    /* ===== CHAT ===== */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        margin: 8px 0 !important;
    }

    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(91, 155, 245, 0.2) !important;
        border-radius: 20px !important;
        padding: 4px !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1) !important;
    }

    [data-testid="stChatInput"] textarea {
        border: none !important;
        background: transparent !important;
        border-radius: 16px !important;
        padding: 10px 14px !important;
        color: inherit !important;
    }

    [data-testid="stChatInput"] textarea:focus { box-shadow: none !important; border: none !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: inherit !important; opacity: 0.4 !important; }

    [data-testid="stChatInput"] button {
        border-radius: 14px !important;
        background: linear-gradient(135deg, #5b9bf5, #8b5cf6) !important;
        border: none !important;
    }

    [data-testid="stChatInput"] button svg { fill: white !important; }

    .stChatFloatingInputContainer {
        background: transparent !important;
        border-top: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding-top: 8px !important;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        border: 1px solid rgba(91, 155, 245, 0.4) !important;
        box-shadow: 0 4px 16px rgba(91, 155, 245, 0.15) !important;
    }

    .new-convo-btn > button {
        background: linear-gradient(135deg, rgba(91, 155, 245, 0.15), rgba(139, 92, 246, 0.15)) !important;
        border: 1px solid rgba(91, 155, 245, 0.25) !important;
        font-weight: 600 !important;
    }

    .danger-btn > button {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.25) !important;
        color: inherit !important;
    }

    .danger-btn > button:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border: 1px solid rgba(239, 68, 68, 0.4) !important;
    }

    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] section {
        border-radius: 12px !important;
        border: 1px dashed rgba(91, 155, 245, 0.3) !important;
        background: rgba(91, 155, 245, 0.03) !important;
    }

    /* ===== EXPANDER ===== */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }

    /* ===== HEADER ===== */
    .app-header { text-align: center; padding: 16px 0 8px 0; }

    .app-header h1 {
        font-size: 2.2em !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #5b9bf5, #8b5cf6) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }

    .app-header p { opacity: 0.5; font-size: 0.95em; }

    .welcome-card {
        background: rgba(91, 155, 245, 0.05) !important;
        border: 1px solid rgba(91, 155, 245, 0.1) !important;
        border-radius: 20px !important;
        padding: 40px !important;
        text-align: center !important;
        margin: 40px auto !important;
        max-width: 600px !important;
        color: inherit !important;
    }

    .welcome-card h3 { color: inherit !important; }

    .upload-success {
        background: rgba(34, 197, 94, 0.08) !important;
        border: 1px solid rgba(34, 197, 94, 0.2) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        color: inherit !important;
        margin: 8px 0 !important;
    }

    .filter-label {
        font-size: 0.75em !important;
        opacity: 0.4 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 4px !important;
    }

    .provider-badge {
        font-size: 0.7em !important;
        opacity: 0.3 !important;
        text-align: center !important;
        margin-top: 8px !important;
    }

    .format-list {
        font-size: 0.75em !important;
        opacity: 0.4 !important;
        margin-top: 4px !important;
    }
</style>
""", unsafe_allow_html=True)


# ---- All supported file extensions for the uploader ----
SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "txt", "md", "json", "yaml", "yml", "xml", "html", "htm"]


# ---- Initialize Components ----
@st.cache_resource
def init_components():
    return DocumentParser(), TextChunker(), get_vector_store(), RAGChain()


parser, chunker, vector_store, rag_chain = init_components()


# ---- Session State ----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "upload_complete" not in st.session_state:
    st.session_state.upload_complete = False
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False


# ---- Sidebar ----
with st.sidebar:
    st.markdown("## ⚖️ Legal Q&A")

    # New Conversation
    st.markdown('<div class="new-convo-btn">', unsafe_allow_html=True)
    if st.button("✨ New Conversation", use_container_width=True):
        rag_chain.clear_memory(st.session_state.session_id)
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ---- DOCUMENTS ----
    st.markdown("### 📁 Documents")

    collections = vector_store.list_collections()
    stats = vector_store.get_stats()

    st.markdown(
        f"""<div class="metric-row">
            <div class="metric-badge">
                <div class="metric-value">{stats['total_collections']}</div>
                <div class="metric-label">Collections</div>
            </div>
            <div class="metric-badge">
                <div class="metric-value">{stats['total_documents']}</div>
                <div class="metric-label">Documents</div>
            </div>
            <div class="metric-badge">
                <div class="metric-value">{stats['total_chunks']}</div>
                <div class="metric-label">Chunks</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if collections:
        for coll_name, docs in collections.items():
            with st.expander(f"📁 {coll_name} ({len(docs)} files)", expanded=False):
                for doc in docs:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(
                            f"""<div class="doc-item">
                                <div class="doc-name">📄 {doc['source']}</div>
                                <div class="doc-meta">{doc['chunks']} chunks · {doc['pages']} pg</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['source']}", help=f"Delete {doc['source']}"):
                            vector_store.delete_document(doc["source"])
                            rag_chain.invalidate_caches()
                            st.rerun()

                if st.button(f"🗑️ Delete '{coll_name}'", key=f"delcoll_{coll_name}", use_container_width=True):
                    vector_store.delete_collection_group(coll_name)
                    rag_chain.invalidate_caches()
                    st.rerun()
    else:
        st.caption("No documents yet.")

    st.divider()

    # ---- UPLOAD ----
    st.markdown("### ➕ Upload New")

    if st.session_state.upload_complete:
        st.markdown('<div class="upload-success">✅ Document processed!</div>', unsafe_allow_html=True)
        if st.button("Upload another", use_container_width=True):
            st.session_state.upload_complete = False
            st.rerun()
    else:
        existing_collections = list(collections.keys()) if collections else []
        collection_options = existing_collections + ["➕ Create new collection..."]

        selected_collection = st.selectbox(
            "Collection",
            collection_options,
            index=0 if existing_collections else len(collection_options) - 1,
        )

        if selected_collection == "➕ Create new collection...":
            new_collection = st.text_input("New collection name", placeholder="e.g., Real Estate")
            collection_name = new_collection.strip()
        else:
            collection_name = selected_collection

        uploaded_files = st.file_uploader(
            "Drop files",
            type=SUPPORTED_UPLOAD_TYPES,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )

        st.markdown(
            '<div class="format-list">PDF, DOCX, TXT, MD, JSON, YAML, XML, HTML</div>',
            unsafe_allow_html=True,
        )

        if uploaded_files:
            can_upload = bool(collection_name)
            if not collection_name:
                st.error("Enter a collection name.")

            if st.button("📤 Process", type="primary", use_container_width=True, disabled=not can_upload):
                total_stored = 0
                errors = []

                progress = st.progress(0, text="Processing files...")

                for idx, uploaded_file in enumerate(uploaded_files):
                    progress.progress(
                        (idx) / len(uploaded_files),
                        text=f"Processing {uploaded_file.name}..."
                    )

                    already_exists = vector_store.document_exists(uploaded_file.name)
                    if already_exists:
                        errors.append(f"'{uploaded_file.name}' already exists (skipped)")
                        continue

                    temp_path = f"data/uploads/{uploaded_file.name}"
                    os.makedirs("data/uploads", exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        doc_pages = parser.parse(temp_path)
                        chunks = chunker.chunk(doc_pages)
                        num_stored = vector_store.add_chunks(
                            chunks, doc_id=uploaded_file.name, collection_name=collection_name,
                        )
                        total_stored += num_stored
                    except Exception as e:
                        errors.append(f"'{uploaded_file.name}': {str(e)}")

                progress.progress(1.0, text="Done!")
                rag_chain.invalidate_caches()

                if errors:
                    for err in errors:
                        st.warning(err)

                if total_stored > 0:
                    st.session_state.upload_complete = True
                    st.rerun()

    st.divider()

    # ---- SEARCH SCOPE ----
    st.markdown("### 🔍 Search Scope")

    search_all = st.checkbox("All documents", value=True, key="search_all")
    selected_sources = []

    if not search_all and collections:
        st.markdown('<div class="filter-label">Collections</div>', unsafe_allow_html=True)

        for coll_name, docs in collections.items():
            coll_checked = st.checkbox(f"📁 {coll_name} ({len(docs)} files)", key=f"filter_coll_{coll_name}")

            if coll_checked:
                for doc in docs:
                    if doc["source"] not in selected_sources:
                        selected_sources.append(doc["source"])
            else:
                for doc in docs:
                    file_checked = st.checkbox(f"  📄 {doc['source']}", key=f"filter_file_{doc['source']}")
                    if file_checked and doc["source"] not in selected_sources:
                        selected_sources.append(doc["source"])

        if selected_sources:
            st.caption(f"Searching {len(selected_sources)} file(s)")

    source_filters = None if search_all or not selected_sources else selected_sources

    st.divider()

    # ---- SETTINGS ----
    st.markdown("### ⚙️ Settings")

    num_sources = st.slider("Context passages", min_value=1, max_value=20, value=5,
        help="Top-matching passages sent to the AI.")

    use_hybrid = st.toggle("Hybrid search", value=True,
        help="Combines semantic + keyword search for better results.")

    st.divider()

    # ---- RESET ALL DATA ----
    st.markdown("### 🔧 Maintenance")

    if not st.session_state.confirm_reset:
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Reset All Data", use_container_width=True):
            st.session_state.confirm_reset = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ This will delete ALL documents, collections, conversations, and caches. This cannot be undone.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm", type="primary", use_container_width=True):
                # Clear vector store
                vector_store.delete_collection()
                # Clear caches
                rag_chain.invalidate_caches()
                # Clear conversation memory
                rag_chain.clear_memory(st.session_state.session_id)
                # Clear uploaded files
                upload_dir = "data/uploads"
                if os.path.exists(upload_dir):
                    shutil.rmtree(upload_dir)
                    os.makedirs(upload_dir)
                # Clear session state
                st.session_state.chat_history = []
                st.session_state.session_id = str(uuid.uuid4())[:8]
                st.session_state.confirm_reset = False
                st.session_state.upload_complete = False
                st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()

    # Provider info
    from configs.settings import settings as app_settings
    st.markdown(
        f'<div class="provider-badge">LLM: {app_settings.llm_provider} · '
        f'Store: {app_settings.vector_store_provider}</div>',
        unsafe_allow_html=True,
    )


# ---- Main Content ----
st.markdown(
    """<div class="app-header">
        <h1>⚖️ Legal Document Q&A</h1>
        <p>Upload legal documents · Organize in collections · Get cited answers</p>
    </div>""",
    unsafe_allow_html=True,
)

if stats["total_chunks"] == 0 and not st.session_state.chat_history:
    st.markdown(
        """<div class="welcome-card">
            <h3>👋 Welcome</h3>
            <p>Upload a legal document using the sidebar to get started.<br>
            Create collections to organize documents by category.</p>
            <p style="opacity:0.4; font-size:0.85em; margin-top:16px;">
                Supported: <strong>PDF, DOCX, TXT, MD, JSON, YAML, XML, HTML</strong>
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 View Sources ({len(message['sources'])})", expanded=False):
                for i, src in enumerate(message["sources"], 1):
                    section_tag = f' > {src.get("section", "")}' if src.get("section") and src["section"] != "Introduction" else ""
                    line_tag = f' · {src.get("line_info", "")}' if src.get("line_info") else ""
                    breadcrumb = src.get("breadcrumb", src["source"])
                    st.markdown(
                        f"""<div class="source-box">
                            <strong>Passage {i}</strong><br>
                            <span style="color: #5b9bf5; font-size: 0.85em;">📍 {breadcrumb}</span><br>
                            <span style="opacity: 0.6; font-size: 0.8em;">Page {src["page"]}{line_tag} · Relevance: {int(src["relevance_score"] * 100)}% · {src.get("search_type", "semantic")}</span><br>
                            <em style="font-size: 0.9em;">{src["text_preview"]}</em>
                        </div>""",
                        unsafe_allow_html=True,
                    )

if prompt := st.chat_input("Ask anything about your documents..."):
    if stats["total_chunks"] == 0:
        st.warning("Upload a document first.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = rag_chain.ask(
                question=prompt,
                k=num_sources,
                session_id=st.session_state.session_id,
                source_filters=source_filters,
                use_hybrid=use_hybrid,
            )

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander(f"📚 View Sources ({result['num_sources']})", expanded=False):
                for i, src in enumerate(result["sources"], 1):
                    section_tag = f' > {src.get("section", "")}' if src.get("section") and src["section"] != "Introduction" else ""
                    line_tag = f' · {src.get("line_info", "")}' if src.get("line_info") else ""
                    breadcrumb = src.get("breadcrumb", src["source"])
                    st.markdown(
                        f"""<div class="source-box">
                            <strong>Passage {i}</strong><br>
                            <span style="color: #5b9bf5; font-size: 0.85em;">📍 {breadcrumb}</span><br>
                            <span style="opacity: 0.6; font-size: 0.8em;">Page {src["page"]}{line_tag} · Relevance: {int(src["relevance_score"] * 100)}% · {src.get("search_type", "semantic")}</span><br>
                            <em style="font-size: 0.9em;">{src["text_preview"]}</em>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
