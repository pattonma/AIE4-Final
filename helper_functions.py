from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import Qdrant

def process_file(file):
    documents = []
    if file.path.endswith(".pdf"):
        loader = PyMuPDFLoader(file.path)
        docs = loader.load()
        documents.extend(docs)
    else:
        loader = TextLoader(file.path)
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