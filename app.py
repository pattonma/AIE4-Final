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
from langchain_community.document_loaders import PyMuPDF
import uuid
import chainlit as cl
import os
from helper_functions import process_file, load_documents_from_url, add_to_qdrant

chat_model = ChatOpenAI(model="gpt-4o-mini")
te3_small = OpenAIEmbeddings(model="text-embedding-3-small")
set_llm_cache(InMemoryCache())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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

@cl.on_chat_start
async def on_chat_start():
    qdrant_client = QdrantClient(url=os.environ["QDRANT_ENDPOINT"], api_key=os.environ["QDRANT_API_KEY"])
    global qdrant_store
    qdrant_store = Qdrant(
        client=qdrant_client,
        collection_name="kai_test_docs",
        embeddings=te3_small
    )

    res = await ask_action()
    await handle_response(res)

    # Load the style guide from the local file system
    style_guide_path = "./public/CoExperiences Writing Style Guide V1 (2024).pdf"
    loader = PyMuPDF(style_guide_path)
    style_guide_docs = loader.load()
    style_guide_text = "\n".join([doc.page_content for doc in style_guide_docs])
    
    retriever = qdrant_store.as_retriever()
    global retrieval_augmented_qa_chain
    retrieval_augmented_qa_chain = (
        {
            "context": itemgetter("question") | retriever, 
            "question": itemgetter("question"),
            "writing_style_guide": lambda _: style_guide_text
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | chat_prompt
        | chat_model
    )

@cl.author_rename
def rename(orig_author: str):
    return "Marketing Assistant"

@cl.on_message
async def main(message: cl.Message, message_type: str):
    if message_type == "url":
        # load the file
        docs = load_documents_from_url(message.content)
        splits = text_splitter.split_documents(docs)
        for i, doc in enumerate(splits):
            doc.metadata["user_upload_source"] = f"source_{i}"
        print(f"Processing {len(docs)} text chunks")

        # Add to the qdrant_store
        qdrant_store.add_documents(
            documents=splits
        )

        await cl.Message(f"Processing `{response.url}` done. You can now ask questions!").send()
    
    else:
        response = retrieval_augmented_qa_chain.invoke({"question": message.content})
        await cl.Message(content=response.content).send()

    res = await ask_action()
    await handle_response(res)


## Chainlit helper functions
async def ask_action():
    res = await cl.AskActionMessage(
        content="Pick an action!",
        actions=[
            cl.Action(name="Question", value="question", label="Ask a question"),
            cl.Action(name="File", value="file", label="Upload a file"),
            cl.Action(name="Url", value="url", label="Upload a URL"),
        ],
    ).send()
    return res

async def handle_response(res):
    if res and res.get("value") == "file":
        files = None
        files = await cl.AskFileMessage(
            content="Please upload a Text or PDF file to begin!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=12,
        ).send()

        file = files[0]
        
        msg = cl.Message(
            content=f"Processing `{file.name}`...", disable_human_feedback=True
        )
        await msg.send()

        # load the file
        docs = process_file(file)
        splits = text_splitter.split_documents(docs)
        for i, doc in enumerate(splits):
            doc.metadata["user_upload_source"] = f"source_{i}"
        print(f"Processing {len(docs)} text chunks")

        # Add to the qdrant_store
        qdrant_store.add_documents(
            documents=splits
        )

        msg.content = f"Processing `{file.name}` done. You can now ask questions!"
        await msg.update()
    
    if res and res.get("value") == "url":
        await cl.Message(content="Submit a url link in the message box below.").send()
   
    if res and res.get("value") == "question":
        await cl.Message(content="Ask away!").send()
