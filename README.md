# 🧠 DocMind — Multi-Agent RAG System

> **PDF intelligence powered by Google ADK · Vertex AI · Gemini 1.5 Flash**  
> Upload any document. Ask anything. Get grounded, cited answers.

[![Cloud Run](https://img.shields.io/badge/Deployed%20on-Cloud%20Run-4285F4?style=flat-square&logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![Vertex AI](https://img.shields.io/badge/Agent%20Engine-Vertex%20AI-34A853?style=flat-square&logo=googlecloud&logoColor=white)](https://cloud.google.com/vertex-ai)
[![Google ADK](https://img.shields.io/badge/Multi--Agent-Google%20ADK-EA4335?style=flat-square&logo=google&logoColor=white)](https://google.github.io/adk-docs/)
[![Python](https://img.shields.io/badge/Python-3.11-FBBC04?style=flat-square&logo=python&logoColor=black)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat-square)](https://langchain.com)

---

## 🔗 Live Demo

| Deployment | Link |
|---|---|
| 🌐 Streamlit App (Cloud Run) | |
| 🤖 Vertex AI Agent Engine | `projects/YOUR_PROJECT/locations/us-central1/reasoningEngines/ID` |

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   DocMind RAG Pipeline                  │
│                                                         │
│  ┌──────────────────┐       ┌────────────────────────┐  │
│  │  retriever_agent │──────▶│    answer_agent        │  │
│  │                  │       │                        │  │
│  │ • FAISS top-k    │       │ • Gemini 1.5 Flash     │  │
│  │ • Semantic search│       │ • Grounded synthesis   │  │
│  │ • Chunk tracing  │       │ • Source citation      │  │
│  └──────────────────┘       └────────────────────────┘  │
│         Google ADK SequentialAgent                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Vertex AI Agent Engine
(Cloud Trace observability enabled)
```

**Pipeline:**  
`PDF` → `PyPDF loader` → `RecursiveCharacterTextSplitter (1000 tok / 200 overlap)` → `Gemini Embeddings` → `FAISS index` → `top-k retrieval` → `Gemini 1.5 Flash answer synthesis` → `Cited response`

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini 1.5 Flash (`gemini-1.5-flash`) |
| **Embeddings** | Google Generative AI Embeddings (`embedding-001`) |
| **Vector Store** | FAISS (local) |
| **RAG Orchestration** | LangChain LCEL |
| **Multi-Agent** | Google ADK (`SequentialAgent`) |
| **Deployment** | Google Cloud Run + Vertex AI Agent Engine |
| **Observability** | Cloud Trace (enabled on Agent Engine) |
| **UI** | Streamlit |

---

## Key Features

- **Multi-agent pipeline** — separate retriever and answer agents via Google ADK, enabling independent optimization and tracing of each stage
- **Grounded responses** — all answers grounded strictly in retrieved document context; hallucinations explicitly blocked in prompt
- **Source citation** — every answer references the specific chunks it was derived from
- **Production deployment** — containerized on Cloud Run with HTTPS endpoint; agent logic deployed to Vertex AI Agent Engine
- **End-to-end tracing** — Cloud Trace observability on agent invocations for latency + cost analysis
- **Chunking strategy** — 1000-token chunks with 200-token overlap, tuned for semantic coherence across paragraph boundaries

---

## Project Structure

```
RAG-Chatbot/
├── app.py                  # Streamlit UI — upload, chat, source display
├── rag_pipeline.py         # Core RAG: load → chunk → embed → retrieve → answer
├── multi_agent_rag.py      # Google ADK two-agent pipeline (retriever + answer)
├── deploy_to_vertex.py     # Vertex AI Agent Engine deployment script
├── Dockerfile              # Cloud Run container (port 8080)
├── requirements.txt        # Pinned dependencies
└── DEPLOY_GUIDE.md         # Step-by-step deployment commands
```

---

## Run Locally

```bash
git clone https://github.com/abhishekksva/RAG-Chatbot.git
cd RAG-Chatbot
pip install -r requirements.txt

# Standard Streamlit app
streamlit run app.py

# OR: Multi-agent ADK version (requires: pip install google-adk)
export GOOGLE_API_KEY=your_key_here
adk web --agent multi_agent_rag
```

---

## Deploy

### Cloud Run (Streamlit app)
```bash
gcloud run deploy docmind \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi
```

### Vertex AI Agent Engine (multi-agent)
```bash
pip install "google-cloud-aiplatform[adk,reasoningengine]"
python deploy_to_vertex.py --project YOUR_PROJECT_ID
```

See [DEPLOY_GUIDE.md](./DEPLOY_GUIDE.md) for full setup instructions.


## Built by

**Abhishek Krishna Srivastava**  
[GitHub](https://github.com/abhishekksva) · [LinkedIn](#) · [Live Demo](#)

---

*Built with LangChain · FAISS · Google Gemini · Google ADK · Vertex AI · Cloud Run*
