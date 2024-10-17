from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Qdrant
import os

def process_file(file_or_url):
    if isinstance(file_or_url, str) and file_or_url.startswith(('http://', 'https://')):
        # Handle URL
        loader = WebBaseLoader(file_or_url)
        docs = loader.load()
        documents.extend(docs)
    # save the file temporarily
    temp_file = "./"+file_or_url.path
    with open(temp_file, "wb") as file:
       file.write(file_or_url.content)
       file_name = file_or_url.name

    documents = []
    if file_or_url.path.endswith(".pdf"):
        loader = PyMuPDFLoader(temp_file)
        docs = loader.load()
        documents.extend(docs)
    else:
        loader = TextLoader(temp_file)
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