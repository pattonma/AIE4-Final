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