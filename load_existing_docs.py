import models
#import constants
#from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore, Qdrant
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from qdrant_client.http.models import VectorParams
import pymupdf
import requests

#qdrant = QdrantVectorStore.from_existing_collection(
#    embedding=models.basic_embeddings,
#    collection_name="kai_test_documents",
#    url=constants.QDRANT_ENDPOINT,
#)

def extract_links_from_pdf(pdf_path):
    links = []
    doc = pymupdf.open(pdf_path)
    for page in doc:
        for link in page.get_links():
            if link['uri']:
                links.append(link['uri'])
    return links

def load_documents_from_url(url):
    try:
        # Check if it's a PDF
        if url.endswith(".pdf"):
            try:
                loader = PyPDFLoader(url)
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
    

#gather kai's docs
filepaths = ["./test_docs/Employee Statistics FINAL.pdf","./test_docs/Employer Statistics FINAL.pdf","./test_docs/Articles To Share.pdf"]

all_links = []
for pdf_path in filepaths:
    all_links.extend(extract_links_from_pdf(pdf_path))

unique_links = list(set(all_links))
print(unique_links)

documents = []
for link in unique_links:
    doc = load_documents_from_url(link)
    #print(f"loaded doc from {link}")
    if doc:
        documents.extend(doc)


#print(len(documents))
semantic_split_docs = models.semanticChunker.split_documents(documents)
RCTS_split_docs = models.RCTS.split_documents(documents)


#for file in filepaths:
#    loader = PyPDFLoader(file)
#    documents = loader.load()
#    for doc in documents:
#        doc.metadata = {
#            "source": file,
#            "tag": "employee" if "employee" in file.lower() else "employer"
#        }
#    all_documents.extend(documents) 

#chunk them
#semantic_split_docs = models.semanticChunker.split_documents(all_documents)


#add them to the existing qdrant client
collection_name = "docs_from_ripped_urls_recursive"

collections = models.qdrant_client.get_collections()
collection_names = [collection.name for collection in collections.collections]
# If the collection does not exist, create it
if collection_name not in collection_names:
    models.qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance="Cosine") 
    )

qdrant_vector_store = QdrantVectorStore(
    client=models.qdrant_client,
    collection_name=collection_name,
    embedding=models.te3_small
)

qdrant_vector_store.add_documents(RCTS_split_docs)



collection_info = models.qdrant_client.get_collection(collection_name)
print(f"Number of points in collection: {collection_info.points_count}")