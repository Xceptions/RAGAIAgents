from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration Constants
SOURCE_DIR = "/opt/airflow/data/raw_docs"
CHROMA_SERVER_HOST = "localhost" 
CHROMA_SERVER_PORT = 8000
COLLECTION_NAME = "scalable_knowledge_base"

default_args = {
    'owner': 'data_engineering',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def ingest_and_process_documents():
    """Reads new files, builds embeddings and syncs vectors directly to ChromaDB."""
    if not os.path.exists(SOURCE_DIR) or not os.listdir(SOURCE_DIR):
        print("No documents found to process.")
        return

    # 1. Initialize self-hosted LangChain embedding adapter using Ollama
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=f"http://localhost:11434"
    )

    # 2. Connect to the distributed external ChromaDB server cluster
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=None, # None forces client-server mode instead of local files
        client_settings=None,
        connection_string=f"http://{CHROMA_SERVER_HOST}:{CHROMA_SERVER_PORT}"
    )

    # 3. Scan and load document pipelines
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    for file_name in os.listdir(SOURCE_DIR):
        file_path = os.path.join(SOURCE_DIR, file_name)
        documents = []
        
        if file_name.endswith('.txt'):
            loader = TextLoader(file_path)
            documents = loader.load()
        elif file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()

        if documents:
            # Chunk split for semantic search optimization
            chunks = text_splitter.split_documents(documents)
            
            # Batch upsert documents to support horizontal scalability
            vector_store.add_documents(chunks)
            print(f"Successfully processed and indexed: {file_name}")
            
            # Move processed documents to archive to prevent duplicate processing
            os.remove(file_path)

with DAG(
    'rag_document_ingestion_pipeline',
    default_args=default_args,
    description='Scalable batch processing pipeline for localized vector injection',
    schedule_interval='@hourly',
    catchup=False
) as dag:

    process_docs_task = PythonOperator(
        task_id='ingest_and_embed_documents',
        python_callable=ingest_and_process_documents,
    )

    process_docs_task
