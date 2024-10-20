import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTRHOPIC_API_KEY = os.environ.get("ANTRHOPIC_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT='https://api.smith.langchain.com'
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")