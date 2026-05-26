from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db" # where we want chroma to store the vectors

# define our chroma vector db with a persistent location
# and storage
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location, # to persist
    embedding_function=embeddings
)


if not os.path.exists(db_location):
    # prepare our reviews as a document
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content = row['Title'] + " " + row['Review'],
            metadata = {"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

    # add the documents prepared above to the chroma db
    vector_store.add_documents(documents=documents, ids=ids)

# to retrieve the data
retriever  = vector_store.as_retriever(
    search_kwargs={"k": 5} # our many documents to retrieve

)