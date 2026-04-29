"""
Deploy DocMind Multi-Agent to Vertex AI Agent Engine
=====================================================
Prerequisites:
  1. gcloud auth login
  2. gcloud config set project YOUR_PROJECT_ID
  3. pip install google-cloud-aiplatform[adk,reasoningengine]

Usage:
  python deploy_to_vertex.py --project YOUR_PROJECT_ID --pdf path/to/sample.pdf
"""

import argparse
import os
import vertexai
from vertexai.preview import reasoning_engines
from multi_agent_rag import rag_pipeline, build_index


def deploy(project_id: str, location: str = "us-central1"):
    print(f"\n🚀 Deploying DocMind to Vertex AI Agent Engine...")
    print(f"   Project: {project_id} | Region: {location}\n")

    vertexai.init(project=project_id, location=location)

    # Wrap the ADK agent in a ReasoningEngine app
    app = reasoning_engines.AdkApp(
        agent=rag_pipeline,
        enable_tracing=True,   # enables Cloud Trace — key FDE observability skill
    )

    # Deploy (this takes ~3-5 minutes)
    remote_app = reasoning_engines.ReasoningEngine.create(
        app,
        requirements=[
            "google-adk>=0.3.0",
            "langchain-community>=0.2.16",
            "langchain-google-genai>=1.0.10",
            "faiss-cpu>=1.8.0",
            "pypdf>=4.3.1",
        ],
        display_name="DocMind RAG Pipeline",
        description="Two-agent RAG pipeline: retriever + answer synthesizer",
    )

    print("\n✅ Deployment successful!")
    print(f"   Resource name: {remote_app.resource_name}")
    print(f"\n   Save this resource name — you'll need it to query the deployed agent.\n")

    return remote_app


def query_deployed(remote_app, question: str):
    """Query the deployed agent on Vertex AI."""
    response = remote_app.query(input=question)
    print(f"\nQ: {question}")
    print(f"A: {response}")
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy DocMind to Vertex AI")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="GCP region")
    parser.add_argument("--pdf", help="Optional: path to PDF to index before deploy test")
    args = parser.parse_args()

    # Optional: build index locally to test before deploy
    if args.pdf:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Set GOOGLE_API_KEY to test locally before deploying.")
        else:
            print(build_index(args.pdf, api_key))

    remote_app = deploy(args.project, args.location)

    # Quick smoke test on deployed agent
    query_deployed(remote_app, "What is this document about?")
