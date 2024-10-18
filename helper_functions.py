from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredURLLoader
from langchain_community.vectorstores import Qdrant
import os
import requests

def process_file(file):
    # save the file temporarily
    temp_file = "./"+file.path
    with open(temp_file, "wb") as file:
       file.write(file.content)

    documents = []
    if file.path.endswith(".pdf"):
        loader = PyMuPDF(temp_file)
        docs = loader.load()
        documents.extend(docs)
    else:
        loader = TextLoader(temp_file)
        docs = loader.load()
        documents.extend(docs)
    return documents

def load_documents_from_url(url):
    try:
        # Check if it's a PDF
        if url.endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(url)
                return loader.load()
            except Exception as e:
                print(f"Error loading PDF from {url}: {e}")
                return None
        
        # Fetch the content and check for video pages
        try:
            response = requests.head(url, timeout=10)  # Timeout for fetching headers
            content_type = response.headers.get('Content-Type', '')
        except Exception as e:
            print(f"Error fetching headers from {url}: {e}")
            return None
        
        # Ignore video content (flagged for now)
        if 'video' in content_type:
            return None
        if 'youtube' in url:
            return None
        
        # Otherwise, treat it as an HTML page
        try:
            loader = UnstructuredURLLoader([url])
            return loader.load()
        except Exception as e:
            print(f"Error loading HTML from {url}: {e}")
            return None
    except Exception as e:
        print(f"General error loading from {url}: {e}")
        return None

def add_to_qdrant(documents, embeddings, qdrant_client, collection_name):
    Qdrant.from_documents(
        documents,
        embeddings,
        url=qdrant_client.url,
        prefer_grpc=True,
        collection_name=collection_name,
    )