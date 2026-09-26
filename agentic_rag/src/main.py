import os
import threading
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError, HttpUrl
from graph import create_agentic_rag_graph
from ingest_urls import fetch_wikipedia_pages_by_urls, chunk_documents, ingest_to_chroma

app = Flask(__name__)

rag_agent = create_agentic_rag_graph()

class ChatRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    urls: list[HttpUrl]

def background_ingestion(urls: list[str]):
    """Runs the ingestion pipeline in a background thread to prevent API timeouts."""
    try:
        print(f"[Background] Starting ingestion for {len(urls)} URLs...")
        raw_docs = fetch_wikipedia_pages_by_urls(urls)
        if raw_docs:
            chunks = chunk_documents(raw_docs)
            ingest_to_chroma(chunks)
            print("[Background] Ingestion completed successfully.")
        else:
            print("[Background] No documents were extracted.")
    except Exception as e:
        print(f"[Background] Ingestion pipeline failed: {str(e)}")

@app.route("/api/v1/chat", methods=["POST"])
def chat():
    """Handles synchronous agentic RAG queries."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON request body"}), 400
        
    try:
        validated_input = ChatRequest(**data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.errors()}), 422

    try:
        inputs = {"query": validated_input.query}
        result = rag_agent.invoke(inputs)
        
        return jsonify({
            "query": validated_input.query,
            "response": result.get("generation", "No response generated."),
            "sources": [doc.metadata.get("source") for doc in result.get("documents", [])]
        }), 200
        
    except Exception as e:
        app.logger.error(f"Agent execution failure: {str(e)}")
        return jsonify({"error": "Internal server error during agent execution"}), 500

@app.route("/api/v1/ingest", methods=["POST"])
def ingest():
    """Triggers asynchronous document extraction and vector database storage."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON request body"}), 400
        
    try:
        validated_input = IngestRequest(**data)
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.errors()}), 422

    str_urls = [str(url) for url in validated_input.urls]

    thread = threading.Thread(target=background_ingestion, args=(str_urls,))
    thread.daemon = True
    thread.start()

    return jsonify({
        "status": "accepted",
        "message": f"Ingestion task kicked off in background for {len(str_urls)} items."
    }), 202

@app.route("/health", methods=["GET"])
def health_check():
    """Liveness probe endpoint."""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is missing.")
    app.run(host="0.0.0.0", port=5000, debug=False)