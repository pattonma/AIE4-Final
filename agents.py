from helper_functions import create_team_agent
from operator import itemgetter
from langchain_core.runnables.passthrough import RunnablePassthrough
import models
import prompts
import tools

search_agent = create_team_agent(
    models.gpt4o,
    [tools.tavily_tool],
    "You are a research assistant who can search for up-to-date info using the tavily search engine.",
    "Search",
    ["Search", "PaperInformationRetriever"]
)

research_agent = create_team_agent(
    models.gpt4o,
    [tools.retrieve_information],
    "You are a research assistant who can provide specific information on the provided paper: 'murthy-loneliness.pdf'. You must only respond with information about the paper related to the request.",
    "PaperInformationRetriever",
    ["Search", "PaperInformationRetriever"]
)

doc_writer_agent = create_team_agent(
    models.gpt4o,
    [tools.write_document, tools.edit_document, tools.read_document],
    "You are an expert writing technical social media posts.",
    "DocWriter",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

note_taking_agent = create_team_agent(
    models.gpt4o,
    [tools.create_outline, tools.read_document],
    "You are an expert senior researcher tasked with writing a social media post outline and taking notes to craft a social media post.",
    "NoteTaker",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

copy_editor_agent = create_team_agent(
    models.gpt4o,
    [tools.write_document, tools.edit_document, tools.read_document],
    "You are an expert copy editor who focuses on fixing grammar, spelling, and tone issues.",
    "CopyEditor",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

voice_editor_agent = create_team_agent(
    models.gpt4o,
    [tools.write_document, tools.edit_document, tools.read_document],
    "You are an expert in crafting and refining the voice and tone of social media posts. You edit the document to ensure it has a consistent, professional, and engaging voice appropriate for social media platforms.",
    "VoiceEditor",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

simple_rag_chain = (
        {
            "context": itemgetter("question") | models.semantic_tuned_retriever, 
            "question": itemgetter("question"),
            "writing_style_guide": lambda _: prompts.style_guide_text
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | prompts.chat_prompt
        | models.gpt4o
    )