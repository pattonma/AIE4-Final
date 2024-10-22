from helper_functions import create_team_agent
from operator import itemgetter
from langchain_core.runnables.passthrough import RunnablePassthrough
import models
import prompts

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