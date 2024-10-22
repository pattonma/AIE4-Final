from typing import Dict, List, TypedDict, Sequence
from langgraph.graph import StateGraph, END
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
import models
import prompts
import json
from operator import itemgetter
from langgraph.errors import GraphRecursionError


#######################################
###     Research Team Components    ###
#######################################
class ResearchState(TypedDict):
    workflow: List[str]
    topic: str
    research_data: Dict[str, str]
    next: str
    message_to_manager: str
    message_from_manager: str

#
#   Reserach Chains and Tools
#
qdrant_research_chain = (
        {"context": itemgetter("topic") | models.compression_retriever, "topic": itemgetter("topic")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompts.research_query_prompt  | models.gpt4o_mini | StrOutputParser(), "context": itemgetter("context")}
    )

tavily_tool = TavilySearchResults(max_results=3)
query_chain = ( prompts.search_query_prompt | models.gpt4o_mini | StrOutputParser() )
tavily_simple = ({"tav_results": tavily_tool} | prompts.tavily_prompt | models.gpt4o_mini | StrOutputParser())
tavily_chain = (
    {"query": query_chain} | tavily_simple
)

research_supervisor_chain = (
    prompts.research_supervisor_prompt | models.gpt4o | StrOutputParser()
)

#
#   Reserach Node Defs
#
def query_qdrant(state: ResearchState) -> ResearchState:
    #print("qdrant node")
    topic = state["topic"]
    result = qdrant_research_chain.invoke({"topic": topic})
    #print(result)
    state["research_data"]["qdrant_results"] = result["response"]
    state['workflow'].append("query_qdrant")
    #print(state['workflow'])

    return state

def web_search(state: ResearchState) -> ResearchState:
    #print("tavily node")
    # Extract the last message as the topic
    topic = state["topic"]
    #print(topic)
    # Get the Qdrant results from the state
    qdrant_results = state["research_data"].get("qdrant_results", "No previous results available.")
    # Run the web search chain
    result = tavily_chain.invoke({
        "topic": topic,
        "qdrant_results": qdrant_results
    })
    #print(result)
    # Update the state with the web search results
    state["research_data"]["web_search_results"] = result
    state['workflow'].append("web_search")
    #print(state['workflow'])
    return state

def research_supervisor(state):
    #print("research supervisor node")
    message_from_manager = state["message_from_manager"]
    collected_data = state["research_data"]
    topic = state['topic']
    supervisor_result = research_supervisor_chain.invoke({"message_from_manager": message_from_manager, "collected_data": collected_data, "topic": topic})
    lines = supervisor_result.split('\n')
    #print(supervisor_result)
    for line in lines:
        if line.startswith('Next Action: '):
            state['next'] = line[len('Next Action: '):].strip()  # Extract the next action content
        elif line.startswith('Message to project manager: '):
            state['message_to_manager'] = line[len('Message to project manager: '):].strip()
    state['workflow'].append("research_supervisor")
    #print(state['workflow'])
    return state

def research_end(state):
    #print("research_end node")
    state['workflow'].append("research_end")
    #print(state['workflow'])
    return state

#######################################
###     Writing Team Components     ###
#######################################
class WritingState(TypedDict):
    workflow: List[str]
    topic: str
    research_data: Dict[str, str]
    draft_posts: Sequence[str]
    final_post: str
    next: str
    message_to_manager: str
    message_from_manager: str
    review_comments: str
    style_checked: bool

#
#   Writing Chains
#
writing_supervisor_chain = (
    prompts.writing_supervisor_prompt | models.gpt4o | StrOutputParser()
)

post_creation_chain = (
    prompts.post_creation_prompt | models.gpt4o_mini | StrOutputParser()
)

post_editor_chain = (
	prompts.post_editor_prompt | models.gpt4o | StrOutputParser()
)

post_review_chain = (
	prompts.post_review_prompt | models.gpt4o | StrOutputParser()
)

#
#   Writing Node Defs
#
def post_creation(state):
    print("post_creation node")
    topic = state['topic']
    drafts = state['draft_posts']
    collected_data = state["research_data"]
    review_comments = state['review_comments']
    results = post_creation_chain.invoke({"topic": topic, "collected_data": collected_data, "drafts": drafts, "review_comments": review_comments})
    state['draft_posts'].append(results)
    state['workflow'].append("post_creation")
    return state

def post_editor(state):
    print("post_editor node")
    current_draft = state['draft_posts'][-1]
    styleguide = prompts.style_guide_text
    review_comments = state['review_comments']
    results = post_editor_chain.invoke({"current_draft": current_draft, "styleguide": styleguide, "review_comments": review_comments})
    state['draft_posts'].append(results)
    state['workflow'].append("post_editor")
    return state

def post_review(state):
    print("post_review node")
    current_draft = state['draft_posts'][-1]
    styleguide = prompts.style_guide_text
    results = post_review_chain.invoke({"current_draft": current_draft, "styleguide": styleguide})
    data = json.loads(results.strip())
    state['review_comments'] = data["Comments on current draft"]
    if data["Draft Acceptable"] == 'Yes':
        state['final_post'] = state['draft_posts'][-1]
    state['workflow'].append("post_review")
    return state

def writing_end(state):
    print("writing_end node")
    state['workflow'].append("writing_end")
    print(state['workflow'])
    return state

def writing_supervisor(state):
    print("writing_supervisor node")
    message_from_manager = state['message_from_manager']
    topic = state['topic']
    drafts = state['draft_posts']
    final_draft = state['final_post']
    review_comments = state['review_comments']
    supervisor_result = writing_supervisor_chain.invoke({"review_comments": review_comments, "message_from_manager": message_from_manager, "topic": topic, "drafts": drafts, "final_draft": final_draft})
    lines = supervisor_result.split('\n')
    print(supervisor_result)
    for line in lines:
        if line.startswith('Next Action: '):
            state['next'] = line[len('Next Action: '):].strip()  # Extract the next action content
        elif line.startswith('Message to project manager: '):
            state['message_to_manager'] = line[len('Message to project manager: '):].strip()
    state['workflow'].append("writing_supervisor")
    return state

#######################################
###  Overarching Graph Components   ###
#######################################
class State(TypedDict):
    workflow: List[str]
    topic: str
    research_data: Dict[str, str]
    draft_posts: Sequence[str]
    final_post: str
    next: str
    user_input: str
    message_to_manager: str
    message_from_manager: str
    last_active_team :str
    next_team: str
    review_comments: str

#
#   Complete Graph Chains
#
overall_supervisor_chain = (
    prompts.overall_supervisor_prompt | models.gpt4o | StrOutputParser()
)

#
#   Complete Graph Node defs
#
def overall_supervisor(state):
    print("overall supervisor node")
    # Implement overall supervision logic
    init_user_query = state["user_input"]
    message_to_manager = state['message_to_manager']
    last_active_team = state['last_active_team']
    supervisor_result = overall_supervisor_chain.invoke({"query": init_user_query, "message_to_manager": message_to_manager, "last_active_team": last_active_team})
    lines = supervisor_result.split('\n')
    print(supervisor_result)
    for line in lines:
        if line.startswith('Next Action: '):
            state['next_team'] = line[len('Next Action: '):].strip()  # Extract the next action content
        elif line.startswith('Extracted Topic: '):
            state['topic'] = line[len('Extracted Topic: '):].strip()  # Extract the next action content
        elif line.startswith('Message to supervisor: '):
            state['message_from_manager'] = line[len('Message to supervisor: '):].strip()  # Extract the next action content
    state['workflow'].append("overall_supervisor")
    print(state['next_team'])
    print(state['workflow'])
    return state

#######################################
###         Graph structures        ###
#######################################

#
#   Reserach Graph Nodes
#
research_graph = StateGraph(ResearchState)
research_graph.add_node("query_qdrant", query_qdrant)
research_graph.add_node("web_search", web_search)
research_graph.add_node("research_supervisor", research_supervisor)
research_graph.add_node("research_end", research_end)
#
#   Reserach Graph Edges
#
research_graph.set_entry_point("research_supervisor")
research_graph.add_edge("query_qdrant", "research_supervisor")
research_graph.add_edge("web_search", "research_supervisor")
research_graph.add_conditional_edges(
    "research_supervisor",
    lambda x: x["next"],
    {"query_qdrant": "query_qdrant", "web_search": "web_search", "FINISH": "research_end"},
)
research_graph_comp = research_graph.compile()

#
#   Writing Graph Nodes
#
writing_graph = StateGraph(WritingState)
writing_graph.add_node("post_creation", post_creation)
writing_graph.add_node("post_editor", post_editor)
writing_graph.add_node("post_review", post_review)
writing_graph.add_node("writing_supervisor", writing_supervisor)
writing_graph.add_node("writing_end", writing_end)
#
#   Writing Graph Edges
#
writing_graph.set_entry_point("writing_supervisor")
writing_graph.add_edge("post_creation", "post_editor")
writing_graph.add_edge("post_editor", "post_review")
writing_graph.add_edge("post_review", "writing_supervisor")
writing_graph.add_conditional_edges(
    "writing_supervisor",
    lambda x: x["next"],
    {"NEW DRAFT": "post_creation", 
     "FINISH": "writing_end"},
)
writing_graph_comp = writing_graph.compile()

#
#   Complete Graph Nodes
#
overall_graph = StateGraph(State)
overall_graph.add_node("overall_supervisor", overall_supervisor)
overall_graph.add_node("research_team_graph", research_graph_comp)
overall_graph.add_node("writing_team_graph", writing_graph_comp)
#
#   Complete Graph Edges
#
overall_graph.set_entry_point("overall_supervisor")
overall_graph.add_edge("research_team_graph", "overall_supervisor")
overall_graph.add_edge("writing_team_graph", "overall_supervisor")
overall_graph.add_conditional_edges(
    "overall_supervisor",
    lambda x: x["next_team"],
    {"research_team": "research_team_graph",
     "writing_team": "writing_team_graph", 
     "FINISH": END},
)
app = overall_graph.compile()


#######################################
###         Run method              ###
#######################################

def getSocialMediaPost(userInput: str) -> str:
    finalPost = ""
    initial_state = State(
        workflow = [],
        topic= "",
        research_data = {},
        draft_posts = [],
        final_post = [],
        next = [],
        next_team = [],
        user_input=userInput,
        message_to_manager="",
        message_from_manager="",
        last_active_team="",
        review_comments=""
    )
    results = app.invoke(initial_state)
    try:
        results = app.invoke(initial_state, {"recursion_limit": 30})
    except GraphRecursionError:
        return "Recursion Error"
    finalPost = results.final_post
    return finalPost