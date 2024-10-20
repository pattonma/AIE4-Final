import chainlit as cl
from helper_functions import process_file, load_documents_from_url, add_to_qdrant
import models
import agents

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
    cl.Message(message.content).send()
    if message.content.startswith("http://") or message.content.startswith("https://"):
        message_type = "url"
    else:
        message_type = "question"
    
    if message_type == "url":
        # load the file
        docs = load_documents_from_url(message.content)
        cl.Message("loaded docs").send()
        splits = models.semanticChunker_tuned.split_documents(docs)
        cl.Message("split docs").send()
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
        splits = models.semanticChunker_tuned.split_documents(docs)
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
