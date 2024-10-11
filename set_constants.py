import constants
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'constants.py')
constantsFile = open(file_path, "w")
constantsFile.write("OPENAI_API_KEY='" + os.getenv("OPENAI_API_KEY") + "';\n");
constantsFile.write("ANTRHOPIC_API_KEY='" + os.getenv("ANTRHOPIC_API_KEY") + "';\n");
constantsFile.write("LANGCHAIN_API_KEY='" + os.getenv("LANGCHAIN_API_KEY") + "';\n");
constantsFile.write("LANGCHAIN_TRACING_V2=True;\n");
constantsFile.write("LANGCHAIN_ENDPOINT='https://api.smith.langchain.com';\n");
constantsFile.write("QDRANT_API_KEY='" + os.getenv("QDRANT_API_KEY") + "';\n");
constantsFile.write("QDRANT_ENDPOINT='" + os.getenv("QDRANT_ENDPOINT") + "';\n");
constantsFile.close()