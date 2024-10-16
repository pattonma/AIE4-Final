from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import Qdrant
import os

def process_file(uploaded_file):
    # save the file temporarily
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
       file.write(uploaded_file.content)
       file_name = uploaded_file.name

    documents = []
    if uploaded_file.path.endswith(".pdf"):
        loader = PyMuPDFLoader(temp_file)
        docs = loader.load()
        documents.extend(docs)
    else:
        loader = TextLoader(tmp_location)
        docs = loader.load()
        documents.extend(docs)
    return documents


def add_to_qdrant(documents, embeddings, qdrant_client, collection_name):
    
    Qdrant.from_documents(
        documents,
        embeddings,
        url=qdrant_client.url,
        prefer_grpc=True,
        collection_name=collection_name,
    )