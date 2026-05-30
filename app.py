import streamlit as st
from rag_engine import (
    parse_pdf,
    chunk_text,
    embed_chunks,
    run_rag_pipeline
)

st.set_page_config(
    page_title="Smart Document Search · Cohere",
    page_icon="🟢",
    layout="wide"
)

# ── COHERE-STYLE CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #eeebe6;
        color: #1a1a1a;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #e8e4de;
        border-right: 1px solid #e5e5e5;
    }

    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }

    /* Hide default streamlit header */
    [data-testid="stHeader"] {
        background-color: #eeebe6;
        border-bottom: 1px solid #e5e5e5;
    }

    /* Main content area */
    .main .block-container {
        padding: 2.5rem 3rem;
        max-width: 900px;
    }

    /* Title */
    h1 {
        font-size: 1.4rem !important;
        font-weight: 500 !important;
        color: #1a1a1a !important;
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem !important;
    }

    /* Subheaders */
    h2, h3 {
        color: #1a1a1a !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }

    /* Caption / muted text */
    .stCaption, caption {
        color: #888880 !important;
        font-size: 0.85rem !important;
    }

    /* Text input */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        color: #1a1a1a !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.95rem !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #39594a !important;
        box-shadow: 0 0 0 2px rgba(57, 89, 74, 0.3) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #555550 !important;
    }

    /* Primary button — Cohere green */
    .stButton > button[kind="primary"] {
        background-color: #2a4a3a !important;
        color: #7ecba1 !important;
        border: 1px solid #39594a !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #334f41 !important;
        border-color: #4a6b57 !important;
    }

    /* Secondary button */
    .stButton > button {
        background-color: #f9f9f9 !important;
        color: #e8e8e3 !important;
        border: 1px solid #d5d5d5 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #faf9f7 !important;
        border: 1px dashed #d5d5d5 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] > div {
        background-color: #faf9f7 !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #faf9f7 !important;
        border: none !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #d5d5d5 !important;
    }

    [data-testid="stFileDropzone"] {
        background-color: #f9f9f9 !important;
        border: 1px dashed #d5d5d5 !important;
    }

    /* Info box */
    .stAlert {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        color: #888880 !important;
    }

    /* Success box */
    .stSuccess {
        background-color: #1a2a20 !important;
        border: 1px solid #2a4a35 !important;
        border-radius: 8px !important;
        color: #7ecba1 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        color: #888880 !important;
        font-size: 0.85rem !important;
    }

    .streamlit-expanderContent {
        background-color: #141414 !important;
        border: 1px solid #2a2a2a !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* Answer card */
    .answer-card {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        color: #e8e8e3;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-top: 1rem;
    }

    /* Chunk card */
    .chunk-card {
        background-color: #141414;
        border: 1px solid #222222;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        font-size: 0.82rem;
        color: #888880;
        line-height: 1.6;
    }

    .chunk-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #7ecba1;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* Pipeline steps in sidebar */
    .pipeline-step {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 0.5rem 0;
        border-bottom: 1px solid #1f1f1f;
        font-size: 0.82rem;
        color: #888880;
    }

    .pipeline-number {
        background-color: #2a4a3a;
        color: #7ecba1;
        border-radius: 4px;
        width: 18px;
        height: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 600;
        flex-shrink: 0;
        margin-top: 1px;
    }

    /* Divider */
    hr {
        border-color: #2a2a2a !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #7ecba1 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0f0f0f; }
    ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; padding: 0.5rem 0 1.5rem;">
            <div style="width:28px; height:28px; background:#2a4a3a; border-radius:6px;
                        display:flex; align-items:center; justify-content:center;
                        font-size:14px;"></div>
            <span style="font-size:1rem; font-weight:500; color:#e8e8e3;">Document Search</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.75rem; color:#555550; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:0.5rem;">Document</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_file:
        file_names = [f.name for f in uploaded_file]
        if st.session_state.doc_name != file_names:
            with st.spinner(f"Indexing {len(uploaded_file)} document(s)..."):
                all_chunks = []
                for file in uploaded_file:
                    raw_text = parse_pdf(file)
                    chunks = chunk_text(raw_text, chunk_size=400, overlap=80)
                    all_chunks.extend(chunks)
                embeddings = embed_chunks(all_chunks)
                st.session_state.chunks = all_chunks
                st.session_state.embeddings = embeddings
                st.session_state.doc_name = file_names
            st.success(f"✅ {len(all_chunks)} chunks indexed across {len(uploaded_file)} file(s)")
        else:
            st.success(f"✅ {len(uploaded_file)} file(s) loaded — {len(st.session_state.chunks)} chunks in memory")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem; color:#555550; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:0.75rem;">Pipeline</p>', unsafe_allow_html=True)

    steps = [
        ("1", "Cohere Embed v3", "Vectorize document chunks"),
        ("2", "Vector Search", "Retrieve top 10 candidates"),
        ("3", "Cohere Rerank", "Re-score to top 3 chunks"),
        ("4", "Command R+", "Generate grounded answer"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
            <div class="pipeline-step">
                <div class="pipeline-number">{num}</div>
                <div>
                    <div style="color:#c8c8c3; font-weight:500; font-size:0.82rem;">{title}</div>
                    <div style="color:#555550; font-size:0.75rem;">{desc}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1>Smart Enterprise Document Search</h1>
        <p style="color:#555550; font-size:0.875rem; margin-top:0.25rem;">
            Upload a PDF · Ask questions · Get grounded answers
        </p>
    </div>
""", unsafe_allow_html=True)

if st.session_state.chunks is None:
    st.markdown("""
        <div style="background:#faf9f7; border:1px dashed #d5d0c8; border-radius:10px;
                    padding: 3rem 2rem; text-align:center; margin-top:2rem;">
            <div style="font-size:1.5rem; margin-bottom:0.75rem;">📄</div>
            <div style="color:#999999; font-size:0.9rem;">Upload a PDF in the sidebar to get started</div>
        </div>
    """, unsafe_allow_html=True)
else:
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("", placeholder="Ask a question about your document...", label_visibility="collapsed")
    with col2:
        search = st.button("Search", type="primary", use_container_width=True)

    if search and query.strip():
        with st.spinner("Searching..."):
            result = run_rag_pipeline(
                query=query,
                chunks=st.session_state.chunks,
                chunk_embeddings=st.session_state.embeddings
            )

        st.markdown('<p style="font-size:0.75rem; color:#555550; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-top:1.5rem;">Answer</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("View source chunks"):
            for i, chunk in enumerate(result["sources"], 1):
                st.markdown(f"""
                    <div class="chunk-card">
                        <div class="chunk-label">Chunk {i}</div>
                        {chunk}
                    </div>
                """, unsafe_allow_html=True)