from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, Qdrant
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
import constants
import os

os.environ["LANGCHAIN_API_KEY"] = constants.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = str(constants.LANGCHAIN_TRACING_V2)
os.environ["LANGCHAIN_ENDPOINT"] = constants.LANGCHAIN_ENDPOINT

set_llm_cache(InMemoryCache())

tracer = LangChainTracer()
callback_manager = CallbackManager([tracer])

########################
### Chat Models      ###
########################

opus3 = ChatAnthropic(
    api_key=constants.ANTRHOPIC_API_KEY, 
    temperature=0, 
    model='claude-3-opus-20240229',
    callbacks=callback_manager
)

sonnet35 = ChatAnthropic(
    api_key=constants.ANTRHOPIC_API_KEY, 
    temperature=0, 
    model='claude-3-5-sonnet-20240620',
    max_tokens=4096,
    callbacks=callback_manager
)

gpt4 = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=constants.OPENAI_API_KEY,
    callbacks=callback_manager
)

gpt4o = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=constants.OPENAI_API_KEY,
    callbacks=callback_manager
)

gpt4o_mini = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=constants.OPENAI_API_KEY,
    callbacks=callback_manager
)

########################
### Embedding Models ###
########################

basic_embeddings = HuggingFaceEmbeddings(model_name="snowflake/snowflake-arctic-embed-l")

tuned_embeddings = HuggingFaceEmbeddings(model_name="CoExperiences/snowflake-l-marketing-tuned")

te3_small = OpenAIEmbeddings(api_key=constants.OPENAI_API_KEY, model="text-embedding-3-small")

#######################
### Text Splitters  ###
#######################

semanticChunker = SemanticChunker(
    te3_small,
    breakpoint_threshold_type="percentile"
)

semanticChunker_tuned = SemanticChunker(
    tuned_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85
)

RCTS = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=25,
    length_function=len,
)

#######################
###  Vector Stores  ###
#######################

qdrant_client = QdrantClient(url=constants.QDRANT_ENDPOINT, api_key=constants.QDRANT_API_KEY)

semantic_Qdrant_vs = QdrantVectorStore(
    client=qdrant_client,
    collection_name="docs_from_ripped_urls",
    embedding=te3_small
)

rcts_Qdrant_vs = QdrantVectorStore(
    client=qdrant_client,
    collection_name="docs_from_ripped_urls_recursive",
    embedding=te3_small
)

semantic_tuned_Qdrant_vs = QdrantVectorStore(
    client=qdrant_client,
    collection_name="docs_from_ripped_urls_semantic_tuned",
    embedding=tuned_embeddings
)

#######################
###  Retrievers     ###
#######################
semantic_tuned_retriever = semantic_tuned_Qdrant_vs.as_retriever(search_kwargs={"k" : 10})

compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=semantic_tuned_retriever
)