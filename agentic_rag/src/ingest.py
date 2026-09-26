import os
import re
from urllib.parse import unquote, urlparse
import wikipediaapi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()

USER_AGENT = os.getenv("USER_AGENT")
def extract_title_from_url(url: str) -> str:
    """Parses a Wikipedia URL to extract the clean article title string."""
    path = urlparse(url).path
    if path.startswith("/wiki/"):
        encoded_title = path.split("/wiki/")[-1]
        return unquote(encoded_title).replace("_", " ")
    return ""

def fetch_wikipedia_pages_by_urls(urls: list[str]) -> list[Document]:
    """Extracts content from a list of Wikipedia URLs into LangChain Documents."""
    print("PARSING URLS AND FETCHING WIKIPEDIA ARTICLES")
    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language='en')
    documents = []

    for url in urls:
        title = extract_title_from_url(url)
        if not title:
            print(f"Could not parse a valid Wikipedia title from URL: {url}. Skipping.")
            continue
            
        page = wiki.page(title)
        if not page.exists():
            print(f"Article '{title}' found in URL does not exist on Wikipedia. Skipping.")
            continue
            
        print(f"Successfully fetched: {page.title}")
        doc = Document(
            page_content=page.summary + "\n\n" + page.text,
            metadata={
                "source": url,
                "title": page.title,
                "category": "Wikipedia Article"
            }
        )
        documents.append(doc)
    return documents

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Splits text documents into chunks optimized for vector search."""
    print("CHUNKING TEXT DOCS")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True 
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} fragments.")
    return chunks

def ingest_to_chroma(chunks: list[Document], db_path: str = "./chroma_db"):
    print(f"INGESTING INTO CHROMADB AT '{db_path}'")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = Chroma(
        collection_name="agentic_rag",
        embedding_function=embeddings,
        persist_directory=db_path
    )
    
    vector_store.add_documents(chunks)
    print("Ingestion complete! The data is now available for your LangGraph agent.")

if __name__ == "__main__":
    wikipedia_urls = [
        "https://en.wikipedia.org/wiki/Naruto",
        "https://en.wikipedia.org/wiki/Bleach",
    ]
    
    raw_docs = fetch_wikipedia_pages_by_urls(wikipedia_urls)
    
    if raw_docs:
        processed_chunks = chunk_documents(raw_docs)
        ingest_to_chroma(processed_chunks)
    else:
        print("No articles were successfully fetched. Pipeline terminated.")
