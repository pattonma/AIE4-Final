from helper_functions import create_team_agent
import models

search_agent = create_team_agent(
    models.gpt4o,
    [tavily_tool],
    "You are a research assistant who can search for up-to-date info using the tavily search engine.",
    "Search",
    ["Search", "PaperInformationRetriever"]
)

research_agent = create_team_agent(
    models.gpt4o,
    [retrieve_information],
    "You are a research assistant who can provide specific information on the provided paper: 'murthy-loneliness.pdf'. You must only respond with information about the paper related to the request.",
    "PaperInformationRetriever",
    ["Search", "PaperInformationRetriever"]
)

doc_writer_agent = create_team_agent(
    models.gpt4o,
    [write_document, edit_document, read_document],
    "You are an expert writing technical social media posts.",
    "DocWriter",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

note_taking_agent = create_team_agent(
    models.gpt4o,
    [create_outline, read_document],
    "You are an expert senior researcher tasked with writing a social media post outline and taking notes to craft a social media post.",
    "NoteTaker",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

copy_editor_agent = create_team_agent(
    models.gpt4o,
    [write_document, edit_document, read_document],
    "You are an expert copy editor who focuses on fixing grammar, spelling, and tone issues.",
    "CopyEditor",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)

voice_editor_agent = create_team_agent(
    models.gpt4o,
    [write_document, edit_document, read_document],
    "You are an expert in crafting and refining the voice and tone of social media posts. You edit the document to ensure it has a consistent, professional, and engaging voice appropriate for social media platforms.",
    "VoiceEditor",
    ["DocWriter", "NoteTaker", "CopyEditor", "VoiceEditor"]
)