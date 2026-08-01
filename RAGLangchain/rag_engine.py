from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class ScalableRAGEngine:
    def __init__(self, chroma_host="localhost", chroma_port=8000, collection_name="scalable_knowledge_base"):
        # Configure localized underlying embedding architecture
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Connect strictly to the master vector store server endpoint
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            connection_string=f"http://{chroma_host}:{chroma_port}"
        )
        
        # Instantiate LLM orchestrator running locally via Ollama
        self.llm = ChatOllama(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.2
        )
        
        self.rag_chain = self._build_rag_chain()

    def _format_docs(self, docs):
        """Converts retrieved segments into uniform text injection segments."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_rag_chain(self):
        """Builds an optimized, predictable, low-latency execution chain."""
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4} # Retrieves top 4 contextual matches
        )

        template = """You are a professional assistant. Answer the user question strictly using the provided context. 
If you do not know the answer based on the context, state clearly that the answer is not available.

Context:
{context}

Question:
{question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        # LCEL (LangChain Expression Language) execution architecture
        chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def query(self, user_question: str) -> str:
        """Executes full search and inference orchestration loop."""
        return self.rag_chain.invoke(user_question)
