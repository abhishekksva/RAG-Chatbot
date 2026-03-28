import streamlit as st
import tempfile
import os
from rag_pipeline import build_rag_pipeline, ask_question

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Chatbot",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0F0F0F;
    color: #F0EDE6;
}

.main { background-color: #0F0F0F; }

.stApp { background-color: #0F0F0F; }

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #F0EDE6 0%, #C8A96E 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    line-height: 1.1;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    color: #888;
    font-size: 1rem;
    margin-top: 4px;
    margin-bottom: 2rem;
    letter-spacing: 0.05em;
}

.step-card {
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.step-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #C8A96E;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.chat-bubble-user {
    background: #C8A96E22;
    border: 1px solid #C8A96E44;
    border-radius: 12px 12px 4px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #F0EDE6;
    font-size: 0.95rem;
}

.chat-bubble-bot {
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 12px 12px 12px 4px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #D4D0C8;
    font-size: 0.95rem;
    line-height: 1.6;
}

.source-tag {
    background: #111;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    font-size: 0.78rem;
    color: #666;
    margin-top: 0.4rem;
    font-family: monospace;
}

.ready-badge {
    display: inline-block;
    background: #1A3A1A;
    border: 1px solid #2A5A2A;
    color: #4CAF50;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-family: 'DM Sans', sans-serif;
}

.stButton > button {
    background: linear-gradient(135deg, #C8A96E, #A8893E) !important;
    color: #0F0F0F !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    color: #F0EDE6 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stFileUploader {
    background: #1A1A1A !important;
    border: 1px dashed #3A3A3A !important;
    border-radius: 12px !important;
}

div[data-testid="stExpander"] {
    background: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 8px !important;
}

hr { border-color: #2A2A2A !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 DocMind</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">RAG-Powered Document Intelligence · Built with LangChain + Gemini</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Session state ─────────────────────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

# ── Step 1: API Key ───────────────────────────────────────────────────────────
st.markdown('<div class="step-card">', unsafe_allow_html=True)
st.markdown('<div class="step-label">Step 01 — Authentication</div>', unsafe_allow_html=True)
api_key = st.text_input(
    "Google Gemini API Key",
    type="password",
    placeholder="Paste your Gemini API key here...",
    help="Get your free key at https://aistudio.google.com"
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Step 2: Upload PDF ────────────────────────────────────────────────────────
st.markdown('<div class="step-card">', unsafe_allow_html=True)
st.markdown('<div class="step-label">Step 02 — Upload Document</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Any PDF — research paper, report, contract, book chapter"
)

if uploaded_file and api_key:
    if st.button("⚡ Process Document"):
        with st.spinner("Building RAG pipeline... chunking → embedding → indexing"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                st.session_state.qa_chain = build_rag_pipeline(tmp_path, api_key)
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []
                st.success(f"✅ Ready! **{uploaded_file.name}** has been indexed.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                os.unlink(tmp_path)

st.markdown('</div>', unsafe_allow_html=True)

# ── Step 3: Chat ──────────────────────────────────────────────────────────────
if st.session_state.qa_chain:
    st.markdown("---")
    st.markdown(f'<span class="ready-badge">● Document loaded: {st.session_state.doc_name}</span>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-label">Step 03 — Ask Anything</div>', unsafe_allow_html=True)

    question = st.text_input(
        "Your question",
        placeholder="What is this document about? Summarize key findings...",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_btn = st.button("Ask →")
    with col2:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    if ask_btn and question:
        with st.spinner("Retrieving from document..."):
            try:
                result = ask_question(st.session_state.qa_chain, question)
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # ── Chat history ──────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("---")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f'<div class="chat-bubble-user">🙋 {chat["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-bot">🧠 {chat["answer"]}</div>', unsafe_allow_html=True)
            with st.expander("View source chunks"):
                for j, src in enumerate(chat["sources"]):
                    st.markdown(f'<div class="source-tag">Chunk {j+1}: {src}...</div>', unsafe_allow_html=True)
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")

else:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#444; padding: 2rem 0;">
        <div style="font-size:2.5rem">📄</div>
        <div style="font-family:'Syne',sans-serif; font-size:1rem; margin-top:0.5rem;">
            Add your API key and upload a PDF to begin
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#333; font-size:0.75rem; font-family:'DM Sans',sans-serif;">
    Built with LangChain · FAISS · Google Gemini · Streamlit
</div>
""", unsafe_allow_html=True)
