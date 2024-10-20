from pathlib import Path
from typing import Annotated, Dict, List, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import prompts
import models
from operator import itemgetter
from langchain_core.runnables.passthrough import RunnablePassthrough

WORKING_DIRECTORY = Path("/tmp/content/data")
WORKING_DIRECTORY.mkdir(parents=True, exist_ok=True)

tavily_tool = TavilySearchResults(max_results=5)

tool_chain = (
        {
            "context": itemgetter("question") | models.semantic_tuned_retriever, 
            "question": itemgetter("question"),
            "writing_style_guide": lambda _: prompts.style_guide_text
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | prompts.chat_prompt
        | models.gpt4o
    )

@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"]
    ):
  """Use Retrieval Augmented Generation to retrieve information about the 'Extending Llama-3’s Context Ten-Fold Overnight' paper."""
  return tool_chain.invoke({"question" : query})

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
