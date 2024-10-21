from typing import Dict, TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
import models
import prompts
from helper_functions import format_docs
from operator import itemgetter

# Define the state structure
class State(TypedDict):
    messages: Sequence[str]
    research_data: Dict[str, str]
    draft_post: str
    final_post: str


# Research Agent Pieces
qdrant_research_chain = (
        {"context": itemgetter("topic") | models.compression_retriever, "topic": itemgetter("topic")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompts.research_query_prompt  | models.gpt4o_mini | StrOutputParser(), "context": itemgetter("context")}
    )

# Web Search Agent Pieces
tavily_tool = TavilySearchResults(max_results=5)
web_search_chain = (
    {
        "topic": itemgetter("topic"),
        "qdrant_results": itemgetter("qdrant_results"),
    }
    | prompts.search_query_prompt
    | models.gpt4o_mini 
    | StrOutputParser()
    | tavily_tool
    | {
        "topic": itemgetter("topic"),
        "qdrant_results": itemgetter("qdrant_results"),
        "search_results": RunnablePassthrough()
    }
    | prompts.summarize_prompt
    | models.gpt4o_mini 
    | StrOutputParser()
)

def query_qdrant(state: State) -> State:
    # Extract the last message as the input
    input_text = state["messages"][-1]

    # Run the chain
    result = qdrant_research_chain.invoke({"topic": input_text})

    # Update the state with the research results
    state["research_data"]["qdrant_results"] = result

    return state

def web_search(state: State) -> State:
    # Extract the last message as the topic
    topic = state["messages"][-1]
    
    # Get the Qdrant results from the state
    qdrant_results = state["research_data"].get("qdrant_results", "No previous results available.")
    
    # Run the web search chain
    result = web_search_chain.invoke({
        "topic": topic,
        "qdrant_results": qdrant_results
    })
    
    # Update the state with the web search results
    state["research_data"]["web_search_results"] = result
    
    return state

def research_supervisor(state):
    # Implement research supervision logic
    return state

def post_creation(state):
    # Implement post creation logic
    return state

def copy_editing(state):
    # Implement copy editing logic
    return state

def voice_editing(state):
    # Implement voice editing logic
    return state

def post_review(state):
    # Implement post review logic
    return state

def writing_supervisor(state):
    # Implement writing supervision logic
    return state

def overall_supervisor(state):
    # Implement overall supervision logic
    return state

# Create the research team graph
research_graph = StateGraph(State)

research_graph.add_node("query_qdrant", query_qdrant)
research_graph.add_node("web_search", web_search)
research_graph.add_node("research_supervisor", research_supervisor)

research_graph.add_edge("query_qdrant", "research_supervisor")
research_graph.add_edge("web_search", "research_supervisor")
research_graph.add_edge("research_supervisor", "query_qdrant")
research_graph.add_edge("research_supervisor", "web_search")
research_graph.add_edge("research_supervisor", END)

research_graph.set_entry_point("research_supervisor")

# Create the writing team graph
writing_graph = StateGraph(State)

writing_graph.add_node("post_creation", post_creation)
writing_graph.add_node("copy_editing", copy_editing)
writing_graph.add_node("voice_editing", voice_editing)
writing_graph.add_node("post_review", post_review)
writing_graph.add_node("writing_supervisor", writing_supervisor)

writing_graph.add_edge("writing_supervisor", "post_creation")
writing_graph.add_edge("post_creation", "copy_editing")
writing_graph.add_edge("copy_editing", "voice_editing")
writing_graph.add_edge("voice_editing", "post_review")
writing_graph.add_edge("post_review", "writing_supervisor")
writing_graph.add_edge("writing_supervisor", END)

writing_graph.set_entry_point("writing_supervisor")

# Create the overall graph
overall_graph = StateGraph(State)

# Add the research and writing team graphs as nodes
overall_graph.add_node("research_team", research_graph)
overall_graph.add_node("writing_team", writing_graph)

# Add the overall supervisor node
overall_graph.add_node("overall_supervisor", overall_supervisor)

overall_graph.set_entry_point("overall_supervisor")

# Connect the nodes
overall_graph.add_edge("overall_supervisor", "research_team")
overall_graph.add_edge("research_team", "overall_supervisor")
overall_graph.add_edge("overall_supervisor", "writing_team")
overall_graph.add_edge("writing_team", "overall_supervisor")
overall_graph.add_edge("overall_supervisor", END)

# Compile the graph
app = overall_graph.compile()