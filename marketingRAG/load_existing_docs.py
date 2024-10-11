import models
import constants
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore, Qdrant
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client.http.models import VectorParams

#qdrant = QdrantVectorStore.from_existing_collection(
#    embedding=models.basic_embeddings,
#    collection_name="kai_test_documents",
#    url=constants.QDRANT_ENDPOINT,
#)


#gather kai's docs
filepaths = ["./test_docs/Employee Statistics FINAL.pdf","./test_docs/Employer Statistics FINAL.pdf"]
all_documents = []
for file in filepaths:
    loader = PyPDFLoader(file)
    documents = loader.load()
    for doc in documents:
        doc.metadata = {
            "source": file,
            "tag": "employee" if "employee" in file.lower() else "employer"
        }
    all_documents.extend(documents) 

#chunk them
semantic_split_docs = models.semanticChunker.split_documents(all_documents)


#add them to the existing qdrant client
collection_name = "kai_test_docs"

collections = models.qdrant_client.get_collections()
collection_names = [collection.name for collection in collections.collections]
# If the collection does not exist, create it
if collection_name not in collection_names:
    models.qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance="Cosine") 
    )

qdrant_vector_store = Qdrant(
    client=models.qdrant_client,
    collection_name=collection_name,
    embeddings=models.te3_small
)

qdrant_vector_store.add_documents(semantic_split_docs)



collection_info = models.qdrant_client.get_collection(collection_name)
print(f"Number of points in collection: {collection_info.points_count}")