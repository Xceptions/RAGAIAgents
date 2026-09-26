import os
from typing import List, TypedDict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class GraphState(TypedDict):
    query: str
    documents: List[Document]
    generation: str
    loop_count: int
    web_fallback: bool

def get_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = Chroma(
        collection_name="agentic_rag",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store
