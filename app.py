import chainlit as cl
from helper_functions import process_file, load_documents_from_url
import models
import agents
import graph
import asyncio

@cl.on_chat_start
async def on_chat_start():
    global qdrant_store
    qdrant_store = models.semantic_tuned_Qdrant_vs

    global retrieval_augmented_qa_chain
    retrieval_augmented_qa_chain = agents.simple_rag_chain

    res = await ask_action()
    await handle_response(res)

@cl.author_rename
def rename(orig_author: str):
    return "AI Assistant"

@cl.on_message
async def main(message: cl.Message):
    # await cl.Message(f"Processing `{message.content}`", disable_human_feedback=True)
    if message.content.startswith("http://") or message.content.startswith("https://"):
        message_type = "url"
    else:
        message_type = "question"

    if message_type == "url":
        await cl.Message(content=f"Processing `{message.content}`", disable_human_feedback=True).send()
        try:
            # Run the document loading and splitting in a thread
            docs = await asyncio.to_thread(load_documents_from_url, message.content)
            await cl.Message(content="loaded docs").send()

            splits = await asyncio.to_thread(models.semanticChunker_tuned.split_documents, docs)
            await cl.Message(content="split docs").send()

            for i, doc in enumerate(splits):
                doc.metadata["user_upload_source"] = f"source_{i}"
            print(f"Processing {len(docs)} text chunks")

            # Add to the qdrant_store asynchronously
            await asyncio.to_thread(qdrant_store.add_documents, splits)

            await cl.Message(f"Processing `{message.content}` done. You can now ask questions!").send()
            

        except Exception as e:
            await cl.Message(f"Error processing the document: {e}").send()
        
        res = await ask_action()
        await handle_response(res)
    else:
        # Handle the question as usual
        await cl.Message(content="Our specialist is working...", disable_human_feedback=True).send()
        #response = await asyncio.to_thread(retrieval_augmented_qa_chain.invoke, {"question": message.content})
        response = await asyncio.to_thread(graph.getSocialMediaPost,  message.content)
        print(response)
        await cl.Message(content=response).send()
        res = await ask_action()
        await handle_response(res)


## Chainlit helper functions
async def ask_action():
    res = await cl.AskActionMessage(
        content="Pick an action!",
        actions=[
            cl.Action(name="Question", value="question", label="Create a post"),
            cl.Action(name="File", value="file", label="Import a file"),
            cl.Action(name="Url", value="url", label="Import a Webpage"),
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
        docs = await asyncio.to_thread(process_file, file)
        await cl.Message(content="loaded docs").send()
        splits = await asyncio.to_thread(models.semanticChunker_tuned.split_documents, docs)
        await cl.Message(content="split docs").send()
        for i, doc in enumerate(splits):
            doc.metadata["user_upload_source"] = f"source_{i}"
        print(f"Processing {len(docs)} text chunks")

        # Add to the qdrant_store
        await asyncio.to_thread(qdrant_store.add_documents, splits)
        await cl.Message(content="added to vs").send()

        await cl.Message(content=f"Processing `{file.name}` done.").send()

        res = await ask_action()
        await handle_response(res)
    
    if res and res.get("value") == "url":
        await cl.Message(content="Submit a url link in the message box below.").send()
   
    if res and res.get("value") == "question":
        await cl.Message(content="Give us your idea!").send()
