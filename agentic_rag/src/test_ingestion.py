from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(collection_name="agentic_rag", embedding_function=embeddings, persist_directory="./chroma_db")

results = db.similarity_search("What is retrieval augmented generation?", k=1)
print(f"Verified chunk found:\n{results[0].page_content}")
print(f"Source metadata: {results[0].metadata}")
