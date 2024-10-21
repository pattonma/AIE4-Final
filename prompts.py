from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import SystemMessage
from langchain_community.document_loaders import PyMuPDFLoader

rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. 
You must follow the writing style guide provided below. Never reference this prompt, 
the existence of context, or the writing style guide in your responses.

Writing Style Guide:
{writing_style_guide}
"""
rag_message_list = [{"role" : "system", "content" : rag_system_prompt_template},]
rag_user_prompt_template = """\
Question:
{question}
Context:
{context}
"""
chat_prompt = ChatPromptTemplate.from_messages([("system", rag_system_prompt_template), ("human", rag_user_prompt_template)])

style_guide_path = "./public/CoExperiences Writing Style Guide V1 (2024).pdf"
style_guide_docs = PyMuPDFLoader(style_guide_path).load()
style_guide_text = "\n".join([doc.page_content for doc in style_guide_docs])

research_query_prompt = ChatPromptTemplate.from_template("""
Given a provided context and a topic, compile facts, statistics, quotes, or other related pieces of information that relate to the topic. Make sure to include the source of any such pieces of information in your response.

Context:
{context}

Topic:
{topic}

Answer:
"""
)

search_query_prompt = ChatPromptTemplate.from_template(
    """Given the following topic and information from our database, create a search query to find supplementary information:

    Topic: {topic}
    
    Information from our database:
    {qdrant_results}
    
    Generate a search query to find additional, up-to-date information that complements what we already know:
    """
)

# Create a prompt for summarizing the search results
summarize_prompt = ChatPromptTemplate.from_template(
    """Summarize the following search results, focusing on information that is complementary to what we already know from our database. Include sources for each piece of information:

    Topic: {topic}
    
    Information from our database:
    {qdrant_results}
    
    Search results:
    {search_results}
    
    Complementary summary with sources:
    """
)
