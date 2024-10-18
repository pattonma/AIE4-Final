#Change to requirements caller
import sys
import subprocess

def run_pip_install():
    packages = [
        "langgraph",
        "langchain",
        "langchain_openai",
        "langchain_experimental",
        "qdrant-client",
        "pymupdf",
        "tiktoken",
        "huggingface_hub",
        "openai",
        "tavily-python"
    ]
    
    package_string = " ".join(packages)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-qU"] + packages)
        print("All required packages have been installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install packages. Please run the following command manually:")
        print(f"%pip install -qU {package_string}")
        sys.exit(1)

# Run pip install
run_pip_install()

import os
import functools
import operator
from typing import Annotated, List, Tuple, Union, Dict, Optional
from typing_extensions import TypedDict
import uuid
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from huggingface_hub import hf_hub_download

# Environment setup
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# CHANGE TO HF DIRECTORY
WORKING_DIRECTORY = Path("/tmp/content/data")
WORKING_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Utility functions
def create_random_subdirectory():
    random_id = str(uuid.uuid4())[:8]
    subdirectory_path = WORKING_DIRECTORY / random_id
    subdirectory_path.mkdir(exist_ok=True)
    return subdirectory_path

def get_current_files():
    try:
        files = [f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*") if f.is_file()]
        return "\n".join(str(f) for f in files) if files else "No files written."
    except Exception:
        return "Unable to retrieve current files."

# Document loading change to upload in HF 
def fetch_hbr_article():
    pdf_path = hf_hub_download(repo_id="your-username/your-repo-name", filename="murthy-loneliness.pdf")
    return PyMuPDFLoader(pdf_path).load()

# Document processing
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(text)
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    length_function=tiktoken_len,
)

docs = fetch_hbr_article()
split_chunks = text_splitter.split_documents(docs)

# Embedding and vector store setup
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name="extending_context_window_llama_3",
)
qdrant_retriever = qdrant_vectorstore.as_retriever()

# RAG setup
RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
openai_chat_model = ChatOpenAI(model="gpt-4o-mini")

rag_chain = (
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    | rag_prompt | openai_chat_model | StrOutputParser()
)

# Tool definitions
@tool
def create_outline(points: List[str], file_name: str) -> str:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"

@tool
def read_document(file_name: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])

@tool
def write_document(content: str, file_name: str) -> str:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"

@tool
def edit_document(file_name: str, inserts: Dict[int, str] = {}) -> str:
    """Edit a document by inserting text at specific line numbers."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    
    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."
    
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)
    return f"Document edited and saved to {file_name}"

@tool
def retrieve_information(query: str):
    """Use Retrieval Augmented Generation to retrieve information about the 'murthy-loneliness' paper."""
    return rag_chain.invoke({"question": query})

# Agent creation helpers
def create_team_agent(llm, tools, system_prompt, agent_name, team_members):
    return create_agent(
        llm,
        tools,
        f"{system_prompt}\nBelow are files currently in your directory:\n{{current_files}}",
        team_members
    )

def create_agent_node(agent, name):
    return functools.partial(agent_node, agent=agent, name=name)

def add_agent_to_graph(graph, agent_name, agent_node):
    graph.add_node(agent_name, agent_node)
    graph.add_edge(agent_name, "supervisor")

def create_team_supervisor(llm, team_description, team_members):
    return create_team_supervisor(
        llm,
        f"You are a supervisor tasked with managing a conversation between the"
        f" following workers: {', '.join(team_members)}. {team_description}"
        f" When all workers are finished, you must respond with FINISH.",
        team_members
    )

def create_team_chain(graph, team_members):
    return (
        functools.partial(enter_chain, members=team_members)
        | graph.compile()
    )

# LLM setup
llm = ChatOpenAI(model="gpt-4-turbo")

# Agent creation
tavily_tool = TavilySearchResults(max_results=5)

search_agent = create_team_agent(
    llm,
    [tavily_tool],
    "You are a research assistant who can search for up-to-date info using the tavily search engine.",
    "Search",
    ["Search", "PaperInformationRetriever"]
)

research_agent = create_team_agent(
    llm,
    [retrieve_information],
    "You are a research assistant who can provide specific information on the provided paper: 'murthy-loneliness.pdf'. You must only respond with information about the paper related to the request.",
    "PaperInformationRetriever",
    ["Search", "PaperInformationRetriever"]
)

doc_writer_agent = create_team_agent(
    llm,
    [write_document, edit_document, read_document],
    "You are an expert writing technical social media posts.",
    "DocWriter",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

note_taking_agent = create_team_agent(
    llm,
    [create_outline, read_document],
    "You are an expert senior researcher tasked with writing a social media post outline and taking notes to craft a social media post.",
    "NoteTaker",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

copy_editor_agent = create_team_agent(
    llm,
    [write_document, edit_document, read_document],
    "You are an expert copy editor who focuses on fixing grammar, spelling, and tone issues.",
    "CopyEditor",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

voice_editor_agent = create_team_agent(
    llm,
    [write_document, edit_document, read_document],
    "You are an expert in crafting and refining the voice and tone of social media posts. You edit the document to ensure it has a consistent, professional, and engaging voice appropriate for social media platforms.",
    "VoiceEditor",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

# Node creation
search_node = create_agent_node(search_agent, "Search")
research_node = create_agent_node(research_agent, "PaperInformationRetriever")
doc_writing_node = create_agent_node(doc_writer_agent, "DocWriter")
note_taking_node = create_agent_node(note_taking_agent, "NoteTaker")
copy_editing_node = create_agent_node(copy_editor_agent, "CopyEditor")
voice_node = create_agent_node(voice_editor_agent, "VoiceEditor")

# Graph creation
research_graph = StateGraph(ResearchTeamState)
add_agent_to_graph(research_graph, "Search", search_node)
add_agent_to_graph(research_graph, "PaperInformationRetriever", research_node)

authoring_graph = StateGraph(DocWritingState)
add_agent_to_graph(authoring_graph, "DocWriter", doc_writing_node)
add_agent_to_graph(authoring_graph, "NoteTaker", note_taking_node)
add_agent_to_graph(authoring_graph, "CopyEditor", copy_editing_node)
add_agent_to_graph(authoring_graph, "VoiceEditor", voice_node)

# Supervisor creation
research_supervisor = create_team_supervisor(
    llm,
    "Given the following user request, determine the subject to be researched and respond with the worker to act next.",
    ["Search", "PaperInformationRetriever"]
)

doc_writing_supervisor = create_team_supervisor(
    llm,
    "Given the following user request, determine which worker should act next. Each worker will perform a task and respond with their results and status.",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

# Graph compilation
research_graph.add_node("supervisor", research_supervisor)
research_graph.set_entry_point("supervisor")
research_chain = create_team_chain(research_graph, research_graph.nodes)

authoring_graph.add_node("supervisor", doc_writing_supervisor)
authoring_graph.set_entry_point("supervisor")
authoring_chain = create_team_chain(authoring_graph, authoring_graph.nodes)

# Meta-supervisor setup
super_graph = StateGraph(State)
super_graph.add_node("Research team", get_last_message | research_chain | join_graph)
super_graph.add_node("SocialMedia team", get_last_message | authoring_chain | join_graph)
super_graph.add_node("supervisor", supervisor_node)

super_graph.add_edge("Research team", "supervisor")
super_graph.add_edge("SocialMedia team", "supervisor")
super_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "SocialMedia team": "SocialMedia team",
        "Research team": "Research team",
        "FINISH": END,
    },
)
super_graph.set_entry_point("supervisor")
super_graph = super_graph.compile()

# Example usage
user_input = input("Enter your request for the social media post: ")

for s in super_graph.stream(
    {
        "messages": [
            HumanMessage(content=user_input)
        ],
    },
    {"recursion_limit": 50},
):
    if "__end__" not in s:
        print(s)
        print("---")