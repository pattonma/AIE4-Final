from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache
from operator import itemgetter
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore, Qdrant
import uuid
import chainlit as cl
import os
from helper_functions import process_file, add_to_qdrant

chat_model = ChatOpenAI(model="gpt-4o-mini")
te3_small = OpenAIEmbeddings(model="text-embedding-3-small")
set_llm_cache(InMemoryCache())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context.
"""
rag_message_list = [{"role" : "system", "content" : rag_system_prompt_template},]
rag_user_prompt_template = """\
Question:
{question}
Context:
{context}
"""
chat_prompt = ChatPromptTemplate.from_messages([("system", rag_system_prompt_template), ("human", rag_user_prompt_template)])

@cl.on_chat_start
async def on_chat_start():
    qdrant_client = QdrantClient(url=os.environ["QDRANT_ENDPOINT"], api_key=os.environ["QDRANT_API_KEY"])
    qdrant_store = Qdrant(
        client=qdrant_client,
        collection_name="kai_test_docs",
        embeddings=te3_small
    )
    retriever = qdrant_store.as_retriever()

    global retrieval_augmented_qa_chain
    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | chat_prompt
        | chat_model
    )

    await cl.Message(content="Ask away!").send()

@cl.author_rename
def rename(orig_author: str):
    return "AI Assistant"

@cl.on_message
async def main(message: cl.Message):
    response = retrieval_augmented_qa_chain.invoke({"question": message.content})
    await cl.Message(content=response.content).send()