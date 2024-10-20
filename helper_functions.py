from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.vectorstores import Qdrant
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
import os
import functools

def process_file(file_or_url):
    if isinstance(file_or_url, str) and file_or_url.startswith(('http://', 'https://')):
        # Handle URL
        loader = WebBaseLoader(file_or_url)
        docs = loader.load()
        documents.extend(docs)
    # save the file temporarily
    temp_file = "./"+file_or_url.path
    with open(temp_file, "wb") as file:
       file.write(file_or_url.content)
       file_name = file_or_url.name

    documents = []
    if file_or_url.path.endswith(".pdf"):
        loader = PyMuPDFLoader(temp_file)
        docs = loader.load()
        documents.extend(docs)
    else:
        loader = TextLoader(temp_file)
        docs = loader.load()
        documents.extend(docs)
    return documents


def add_to_qdrant(documents, embeddings, qdrant_client, collection_name):
    Qdrant.from_documents(
        documents,
        embeddings,
        url=qdrant_client.url,
        prefer_grpc=True,
        collection_name=collection_name,
    )

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

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

def enter_chain(message: str, members: List[str]):
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }
    return results

def create_team_chain(graph, team_members):
    return (
        functools.partial(enter_chain, members=team_members)
        | graph.compile()
    )

def create_agent(
    llm: BaseLanguageModel,
    tools: list,
    system_prompt: str,
) -> str:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += ("\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason! You are one of the following team members: {{team_members}}.")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor