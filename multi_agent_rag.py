"""
DocMind Multi-Agent RAG — Google ADK Edition
=============================================
Two-agent pipeline built on top of your existing rag_pipeline.py:
  Agent 1 (retriever_agent): Fetches relevant chunks from FAISS
  Agent 2 (answer_agent):    Synthesizes a grounded, cited answer

Run locally:  adk web
Deploy:       See deploy_to_vertex.py
"""

import os
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import FunctionTool
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Global FAISS store (populated at runtime) ─────────────────────────────────
_vectorstore = None


def build_index(pdf_path: str, api_key: str) -> str:
    """
    Builds a FAISS index from a PDF.
    Call this once before running the agent pipeline.
    Returns a status string.
    """
    global _vectorstore
    os.environ["GOOGLE_API_KEY"] = api_key

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    _vectorstore = FAISS.from_documents(chunks, embeddings)

    return f"Index built: {len(chunks)} chunks from {pdf_path}"


# ── Tool: retrieve context ─────────────────────────────────────────────────────
def retrieve_context(query: str) -> str:
    """
    Retrieves the top-4 most relevant document chunks for a given query.
    Uses the FAISS index built from the uploaded PDF.
    Returns concatenated chunk text with chunk numbers for traceability.
    """
    if _vectorstore is None:
        return "ERROR: No document indexed yet. Call build_index() first."

    docs = _vectorstore.similarity_search(query, k=4)

    if not docs:
        return "No relevant context found in the document."

    chunks = []
    for i, doc in enumerate(docs):
        chunks.append(f"[Chunk {i+1}]\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)


retriever_tool = FunctionTool(func=retrieve_context)


# ── Agent 1: Retriever ────────────────────────────────────────────────────────
retriever_agent = Agent(
    name="retriever_agent",
    model="gemini-2.0-flash",
    description=(
        "Specialist in semantic retrieval. Given a user query, "
        "uses the retrieve_context tool to fetch the most relevant "
        "chunks from the indexed document."
    ),
    tools=[retriever_tool],
    instruction="""
You are a precise retrieval specialist. Your ONLY job is to:
1. Receive a user query
2. Call the retrieve_context tool with that query
3. Return the raw retrieved chunks exactly as returned — do not summarize or modify them

Always call the tool. Never answer from memory.
""",
)


# ── Agent 2: Answer Synthesizer ───────────────────────────────────────────────
answer_agent = Agent(
    name="answer_agent",
    model="gemini-2.0-flash",
    description=(
        "Specialist in synthesizing grounded, accurate answers "
        "from retrieved document context."
    ),
    instruction="""
You are an expert at synthesizing clear, accurate answers grounded strictly in document context.

You will receive:
- The original user query
- Retrieved document chunks (from the retriever agent)

Your job:
1. Read the chunks carefully
2. Answer the query using ONLY information present in the chunks
3. If the answer isn't in the chunks, say: "I couldn't find this in the document."
4. Always cite which chunk(s) your answer came from (e.g., "According to Chunk 2...")
5. Be concise but complete

Never hallucinate. Only use what's in the provided context.
""",
)


# ── Sequential Pipeline: Retrieve → Answer ────────────────────────────────────
rag_pipeline = SequentialAgent(
    name="docmind_rag_pipeline",
    sub_agents=[retriever_agent, answer_agent],
    description=(
        "DocMind two-agent RAG pipeline: "
        "retriever_agent fetches context, answer_agent synthesizes a grounded answer."
    ),
)


# ── Entry point for adk web ───────────────────────────────────────────────────
# ADK looks for a variable named `root_agent`
root_agent = rag_pipeline


# ── CLI test (optional) ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY environment variable first.")
        sys.exit(1)

    # Quick smoke test with a sample PDF
    pdf_path = "sample.pdf"  # replace with your PDF
    print(build_index(pdf_path, api_key))
    print("\nIndex ready. Run: adk web")
