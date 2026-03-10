"""
Legal Document Q&A — Streamlit Frontend

A chat-based UI for uploading legal documents and asking questions
with cited answers. Modern glassmorphism design.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.retrieval.vector_store import VectorStore
from src.chains.rag_chain import RAGChain


# ---- Page Config ----
st.set_page_config(
    page_title="Legal Document Q&A",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Modern Glassmorphism CSS ----
st.markdown("""
<style>
    /* ===== FULL WIDTH LAYOUT ===== */
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 1rem !important;
    }

    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        min-width: 320px !important;
        max-width: 360px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* ===== SOURCE BOXES (Dark + Light mode) ===== */
    .source-box {
        background: rgba(100, 140, 200, 0.1) !important;
        border-left: 3px solid #5b9bf5 !important;
        padding: 14px 16px !important;
        margin: 8px 0 !important;
        border-radius: 0 12px 12px 0 !important;
        color: inherit !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
    }

    .source-box strong {
        color: #5b9bf5 !important;
    }

    .source-box em {
        color: inherit !important;
        opacity: 0.85;
    }

    /* ===== GLASSMORPHISM CARDS ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        margin: 10px 0 !important;
    }

    /* ===== DOCUMENT CARD IN SIDEBAR ===== */
    .doc-item {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        margin: 6px 0 !important;
        color: inherit !important;
        transition: background 0.2s ease !important;
    }

    .doc-item:hover {
        background: rgba(255, 255, 255, 0.1) !important;
    }

    .doc-item .doc-name {
        font-weight: 600 !important;
        font-size: 0.9em !important;
        color: inherit !important;
    }

    .doc-item .doc-meta {
        font-size: 0.75em !important;
        opacity: 0.6 !important;
        color: inherit !important;
    }

    /* ===== METRIC BADGES ===== */
    .metric-row {
        display: flex;
        gap: 12px;
        margin: 12px 0;
    }

    .metric-badge {
        background: rgba(91, 155, 245, 0.1) !important;
        border: 1px solid rgba(91, 155, 245, 0.2) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        flex: 1 !important;
        text-align: center !important;
        color: inherit !important;
    }

    .metric-badge .metric-value {
        font-size: 1.5em !important;
        font-weight: 700 !important;
        color: #5b9bf5 !important;
    }

    .metric-badge .metric-label {
        font-size: 0.75em !important;
        opacity: 0.6 !important;
        color: inherit !important;
        margin-top: 2px !important;
    }

    /* ===== CHAT MESSAGES ===== */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        margin: 8px 0 !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
    }

    /* ===== CHAT INPUT AREA ===== */
    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(91, 155, 245, 0.2) !important;
        border-radius: 20px !important;
        padding: 2px !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1) !important;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: none !important;
        background: transparent !important;
        font-size: 1em !important;
        padding: 8px 12px !important;
        color: inherit !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        box-shadow: none !important;
        border: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: inherit !important;
        opacity: 0.4 !important;
    }

    [data-testid="stChatInput"] button {
        border-radius: 14px !important;
        background: linear-gradient(135deg, #5b9bf5, #8b5cf6) !important;
        border: none !important;
        margin: 4px !important;
    }

    [data-testid="stChatInput"] button:hover {
        box-shadow: 0 4px 16px rgba(91, 155, 245, 0.3) !important;
    }

    [data-testid="stChatInput"] button svg {
        fill: white !important;
    }

    /* ===== BOTTOM CHAT INPUT CONTAINER ===== */
    .stChatFloatingInputContainer {
        background: transparent !important;
        border-top: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding-top: 12px !important;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        border: 1px solid rgba(91, 155, 245, 0.4) !important;
        box-shadow: 0 4px 16px rgba(91, 155, 245, 0.15) !important;
    }

    /* ===== NEW CONVERSATION BUTTON ===== */
    .new-convo-btn > button {
        background: linear-gradient(135deg, rgba(91, 155, 245, 0.15), rgba(139, 92, 246, 0.15)) !important;
        border: 1px solid rgba(91, 155, 245, 0.25) !important;
        color: inherit !important;
        font-weight: 600 !important;
    }

    .new-convo-btn > button:hover {
        background: linear-gradient(135deg, rgba(91, 155, 245, 0.25), rgba(139, 92, 246, 0.25)) !important;
    }

    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] section {
        border-radius: 12px !important;
        border: 1px dashed rgba(91, 155, 245, 0.3) !important;
        background: rgba(91, 155, 245, 0.03) !important;
    }

    /* ===== EXPANDER (SOURCES) ===== */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }

    /* ===== HEADER ===== */
    .app-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }

    .app-header h1 {
        font-size: 2.2em !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #5b9bf5, #8b5cf6) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin-bottom: 4px !important;
    }

    .app-header p {
        opacity: 0.5;
        font-size: 0.95em;
    }

    /* ===== WELCOME CARD ===== */
    .welcome-card {
        background: rgba(91, 155, 245, 0.05) !important;
        border: 1px solid rgba(91, 155, 245, 0.1) !important;
        border-radius: 20px !important;
        padding: 40px !important;
        text-align: center !important;
        margin: 40px auto !important;
        max-width: 600px !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        color: inherit !important;
    }

    .welcome-card h3 {
        color: inherit !important;
    }

    /* ===== UPLOAD SUCCESS CARD ===== */
    .upload-success {
        background: rgba(34, 197, 94, 0.08) !important;
        border: 1px solid rgba(34, 197, 94, 0.2) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        color: inherit !important;
        margin: 8px 0 !important;
    }

    /* ===== SETTINGS SECTION ===== */
    .settings-label {
        font-size: 0.8em !important;
        opacity: 0.5 !important;
        margin-bottom: 2px !important;
    }
</style>
""", unsafe_allow_html=True)


# ---- Initialize Components (cached) ----
@st.cache_resource
def init_components():
    parser = DocumentParser()
    chunker = TextChunker()
    vector_store = VectorStore()
    rag_chain = RAGChain()
    return parser, chunker, vector_store, rag_chain


parser, chunker, vector_store, rag_chain = init_components()


# ---- Session State ----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "upload_complete" not in st.session_state:
    st.session_state.upload_complete = False


# ---- Sidebar ----
with st.sidebar:
    st.markdown("## ⚖️ Legal Q&A")

    # New Conversation — at the top
    st.markdown('<div class="new-convo-btn">', unsafe_allow_html=True)
    if st.button("✨ New Conversation", use_container_width=True):
        rag_chain.clear_memory(st.session_state.session_id)
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Document section
    st.markdown("### 📁 Documents")

    documents = vector_store.list_documents()
    stats = vector_store.get_stats()

    # Metrics
    st.markdown(
        f"""<div class="metric-row">
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

    # Document list
    if documents:
        for doc in documents:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"""<div class="doc-item">
                        <div class="doc-name">📄 {doc['source']}</div>
                        <div class="doc-meta">{doc['chunks']} chunks · {doc['pages']} page(s)</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown("<div style='padding-top:8px'></div>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_{doc['source']}", help=f"Delete {doc['source']}"):
                    vector_store.delete_document(doc["source"])
                    st.rerun()
    else:
        st.caption("No documents yet.")

    st.divider()

    # File upload — only show uploader if not just completed an upload
    st.markdown("### ➕ Upload New")

    if st.session_state.upload_complete:
        st.markdown(
            '<div class="upload-success">✅ Document processed successfully!</div>',
            unsafe_allow_html=True,
        )
        if st.button("Upload another", use_container_width=True):
            st.session_state.upload_complete = False
            st.rerun()
    else:
        uploaded_file = st.file_uploader(
            "Drop a file here",
            type=["pdf", "docx", "txt"],
            help="PDF, DOCX, or TXT",
            label_visibility="collapsed",
        )

        if uploaded_file:
            already_exists = vector_store.document_exists(uploaded_file.name)

            if already_exists:
                st.warning(f"'{uploaded_file.name}' exists.")
                overwrite = st.checkbox("Overwrite?", key="overwrite")
            else:
                overwrite = False

            if st.button("📤 Process", type="primary", use_container_width=True):
                with st.spinner("Parsing & embedding..."):
                    temp_path = f"data/uploads/{uploaded_file.name}"
                    os.makedirs("data/uploads", exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        if already_exists and overwrite:
                            vector_store.delete_document(uploaded_file.name)
                        elif already_exists and not overwrite:
                            st.error("Check 'Overwrite' to replace.")
                            st.stop()

                        doc_pages = parser.parse(temp_path)
                        chunks = chunker.chunk(doc_pages)
                        num_stored = vector_store.add_chunks(chunks, doc_id=uploaded_file.name)

                        st.session_state.upload_complete = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    st.divider()

    # Search settings
    st.markdown("### ⚙️ Settings")

    filter_options = ["All documents"] + [doc["source"] for doc in documents]
    selected_filter = st.selectbox(
        "Restrict answers to",
        filter_options,
    )
    source_filter = None if selected_filter == "All documents" else selected_filter

    st.markdown(
        '<div class="settings-label">Context passages sent to the LLM</div>',
        unsafe_allow_html=True,
    )
    num_sources = st.slider(
        "Context passages",
        min_value=1,
        max_value=20,
        value=5,
        help="How many of the top-matching passages to include as context. "
             "The system always searches ALL chunks — this controls how many "
             "of the best matches get sent to the AI for answering. Higher = "
             "more context but slower and potentially noisier.",
        label_visibility="collapsed",
    )


# ---- Main Content ----
st.markdown(
    """<div class="app-header">
        <h1>⚖️ Legal Document Q&A</h1>
        <p>Upload legal documents · Ask questions · Get cited answers</p>
    </div>""",
    unsafe_allow_html=True,
)

# Welcome state
if stats["total_chunks"] == 0 and not st.session_state.chat_history:
    st.markdown(
        """<div class="welcome-card">
            <h3>👋 Welcome</h3>
            <p>Upload a legal document using the sidebar to get started.<br>
            Supported formats: <strong>PDF, DOCX, TXT</strong></p>
            <p style="opacity:0.4; font-size:0.85em; margin-top:16px;">
                Try uploading a lease agreement, privacy policy, or terms of service.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

# Chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 View Sources ({len(message['sources'])})", expanded=False):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(
                        f"""<div class="source-box">
                            <strong>Passage {i}</strong> — {src["source"]}, 
                            Page {src["page"]} 
                            (relevance: {src["relevance_score"]})<br>
                            <em>{src["text_preview"]}</em>
                        </div>""",
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("💬 Ask anything about your documents..."):
    if stats["total_chunks"] == 0:
        st.warning("⚠️ Upload a document first using the sidebar.")
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
                source_filter=source_filter,
            )

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander(f"📚 View Sources ({result['num_sources']})", expanded=False):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(
                        f"""<div class="source-box">
                            <strong>Passage {i}</strong> — {src["source"]}, 
                            Page {src["page"]} 
                            (relevance: {src["relevance_score"]})<br>
                            <em>{src["text_preview"]}</em>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })