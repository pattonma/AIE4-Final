from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
import constants
import os

os.environ["LANGCHAIN_API_KEY"] = constants.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = str(constants.LANGCHAIN_TRACING_V2)
os.environ["LANGCHAIN_ENDPOINT"] = constants.LANGCHAIN_ENDPOINT

tracer = LangChainTracer()
callback_manager = CallbackManager([tracer])

qdrant_client = QdrantClient(url=constants.QDRANT_ENDPOINT, api_key=constants.QDRANT_API_KEY)

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

basic_embeddings = HuggingFaceEmbeddings(model_name="snowflake/snowflake-arctic-embed-l")
#hkunlp_instructor_large = HuggingFaceInstructEmbeddings(
#    model_name = "hkunlp/instructor-large",
#    query_instruction="Represent the query for retrieval: "
#)

te3_small = OpenAIEmbeddings(api_key=constants.OPENAI_API_KEY, model="text-embedding-3-small")

semanticChunker = SemanticChunker(
    te3_small,
    breakpoint_threshold_type="percentile"
)

RCTS = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=25,
    length_function=len,
)