# DocMind — RAG Chatbot 

## Overview
An end-to-end Retrieval-Augmented Generation (RAG) chatbot that lets users upload any PDF document and ask natural language questions. Built with LangChain, Google Gemini, and FAISS vector database.

## Architecture

PDF Upload → Text Chunking → Embeddings (Gemini) → FAISS Index → Retrieval → LLM Answer


## Tech Stack
- **LangChain** — RAG orchestration pipeline
- **Google Gemini 1.5 Flash** — LLM for answer generation
- **FAISS** — Vector similarity search
- **Streamlit** — Interactive web UI
- **PyPDF** — PDF parsing

## Features
- Upload any PDF document
- Automatic chunking and embedding
- Context-aware Q&A with source citations
- Chat history with retrievable source chunks
- Clean dark-themed UI

## How to Run

### 1. Clone the repo

git clone https://github.com/abhishekksva/DocMind-RAG-Chatbot.git
cd DocMind-RAG-Chatbot


### 2. Install dependencies

pip install -r requirements.txt


### 3. Get Gemini API Key
- Go to https://aistudio.google.com
- Create a free API key

### 4. Run the app

streamlit run app.py

## Results
- Accurate document Q&A with source attribution
- Handles research papers, reports, contracts, books
- Sub-5 second response time on standard PDFs

## Live Demo
[Streamlit App Link]
