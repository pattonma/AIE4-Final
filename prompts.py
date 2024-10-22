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
    """Given the following topic and information from our database, create a search query to find supplementary information about the topic, including facts, statistics, surveys, articles, and quotes:

    Topic: {topic}
    
    Information from our database:
    {qdrant_results}
    
    Generate a search query to find additional, up-to-date information that complements what we already know. Only include the query you generate, do not include any extraneous language.

    Output:
    Query
    """
)

tavily_prompt = ChatPromptTemplate.from_template(
    """Summarize the query results into the following format:

    Query results: {tav_results}
    
    Output: 
    Search Result 1 source: ...
    Search Result 1 summary: ...
    Search Result 2 source: ...
    Search Result 2 summary: ...
    ...
    """
)

research_supervisor_prompt = ChatPromptTemplate.from_template(
    """You are the supervisor of a team of researchers with different roles. 
    
    Your team's task is to collect information about a given topic.

    As you are the supervisor, you must determine which of your team members should perform their task next.

    Your team members include 'query_qdrant', which can search a database of information that will relate to the topic, and 'web_search', which can search the wider internet for more information about the topic not contained in our database.

    The available actions are the names of your team members (to indicate which team member should run next), or 'FINISH' to indicate that sufficient data has been collected about the topic, and you wish to send your team's work back to your own supervisor.

    You will have access to the data that the team has collected so far, if any exists. You will use this to determine if any further collection is needed.

    You can send a specific message or request to your supervisor when your next action is to send it back to them. When you are called to perform your own task by the project manager, they may have a helpful message or request of you.

    Message from project manager: {message_from_manager}
    Topic: {topic}
    Data collected so far: {collected_data}

    Output: 
    Next Action: ...
    Message to project manager: ...
    """
)

overall_supervisor_prompt = ChatPromptTemplate.from_template(
    """You are the project manager of two teams who are working to create factually accurate and emotionally meaningful social media posts regarding emotional health in the workplace.
    
    Your task is to coordinate the two teams. The first team, 'research_team', is tasked with collecting information and data about the specific topic that the user asks about. The second team, 'writing_team', will use that collected data and information to generaete posts, and then review and iterate over them until the post is of sufficient quality.
    Each team is headed by its own supervisor, who will manage the individual members of that team.

    As you are the project manager, you will deal directly with the user and their initial query, so if there is not an existing topic that has been extracted from the user's query, you must analyze the query and determine what topic your team will be dealing with.

    After determining the topic, or if a specific topic has already been supplied, you must determine which of the teams should perform their task next.

    The available actions are the names of your teams (to indicate which team should run next), or 'FINISH' to indicate that the writing team has sent you a sufficient quality social media post draft that is ready to share with the user.

    To help you determine which team should run next, or if the post is ready to share with the user, you will have access to the latest team supervisor requests and messages, as well as a flag indicating which team was most recently active.

    If you have to pass along a message to the next team, such as if the writing team had asked that the research team collect more information, or if you want to give specific direction to a supervisor, you can do so.

    If the writing team has given you a quality final post draft, it will be suplied to you as well.

    User query: {query}
    Team message: {message_to_manager}
    Last active team: {last_active_team}
    Final post: {final_post}

    Output: 
    Extracted Topic: ...
    Next Action: ...
    Message to supervisor: ...
    """
)

writing_supervisor_prompt = ChatPromptTemplate.from_template(
    """You are the supervisor of a team of writers and editors. 
    
    Your team's task is to construct a high quality draft of a social media post about the topic, using information and data that has been compiled for you.

    Your team members include:
    1) 'post_creation' - This team member is in charge of using the collected data to draft the initial social media posts.
    2) 'post_editor' - This team member takes the latest draft and ensures that it fits the tone and style requirements set by a writing guide, editting the post if required.
    3) 'post_review' - This team member reviews the latest draft, ensuring all quality checks are met, sources are present, and determines if it meets the standards of the final post.

    Your available actions are 'NEW DRAFT' to indicate that you want your team to compile a new draft post, or 'FINISH' to indicate that a sufficient final post has been created, and you wish to send your team's work back to your own supervisor.

    You will have access to drafts that your team has created so far, as well as the final draft, if either exist. You will use the presence of the final post to determine if your team is finished. If you do not have a final post given to you, you cannot turn your team's work in to your supervisor.

    You will also be given the latest comments from the post_review team member regarding the current status of the draft, including if anything needs changed or not. 

    If you do not have a final post given to you, you cannot turn your team's work in to your supervisor.

    You can send a specific message or request to your supervisor when your next action is to send it back to them. When you are called to perform your own task by the project manager, they may have a helpful message or request of you.

    Message from project manager: {message_from_manager}
    Topic: {topic}
    Drafts so far: {drafts}
    Final Post: {final_draft}
    Comments from Reviewer: {review_comments}

    Output: 
    Next Action: ...
    Message to project manager: ...
    """
)

post_creation_prompt = ChatPromptTemplate.from_template(
    """You are a writer on a team tasked with creating professional social media posts about mental health in the workplace.

    You are tasked with creating drafts of posts that a marketing team can use either as they are, or as bases for posts on various social media sites.

    You will be given a set of collected data and information that are related to the market team's current post idea, including but not limited to facts, statistics, quotes, articles, surveys, etc. This data will include sources.

    You will also be given a set of previous drafts of the post, if any exist, so that you can iterate and improve on what was created previously, as well as the latest set of comments from the post_review team member, if there are any. These comments will help you improve over previous drafts.

    The previous review comments will outline what was satisfactory about the last draft and what was not satisfactory. Take the comments into account to ensure that you do not make the same mistakes on the current draft.

    Using the data, write a draft of a social media post about the topic. Do not make up new facts or data to base your post on, use only the data provided. Where possible, also include the source of the information you use in the post.

    The post should not be more than 200 words.

    Topic: {topic}
    Drafts so far: {drafts}
    Collected data: {collected_data}
    Comments from previous review: {review_comments}

    Output:
    (Draft post)
    """
)

post_editor_prompt = ChatPromptTemplate.from_template(
    """You are a writer on a team tasked with creating professional social media posts about mental health in the workplace.

    You are tasked with ensuring that a current draft of a social media post follows your organization's style, tone, and voice guidelines as dictated by a style guide. 

    You will be given the current draft of the post, the organization's writing style guide to review, and any comments from the previous draft's review.

    The previous review comments will outline what was satisfactory about the last draft and what was not satisfactory. Take the comments into account to ensure that you do not make the same mistakes on the current draft.

    You are to rewrite the draft post, if required, to adhere the writing style and professional expectations outlined in the style guide. You **MUST** adhere to the style guide.
    The underlying message and content of the post should remain unchanged. All factual information, quotes, or sources included in the post should be included in the rewrite.

    Your output should not include any of your supplementary comments.

    Current draft: {current_draft}
    Style guide: {styleguide}
    Comments from previous review: {review_comments}

    Output:
    (Updated draft)
    """
)

post_review_prompt = ChatPromptTemplate.from_template(
    """You are a writer on a team tasked with creating professional social media posts about mental health in the workplace.

    You are tasked with determining if a current draft social media post is of sufficient quality to become the final draft.

    You will be given the current draft of the post, and the organization's writing style guide.

    Review the draft's content, overall message, and general adherence to the writing guide. Focus on major issues that significantly impact the post's effectiveness or professionalism. Minor stylistic variations should not be grounds for rejection.

    Consider the following when reviewing:
    1. Does the post convey its main message clearly?
    2. Is the tone appropriate for a professional social media post about mental health in the workplace?
    3. Are there any major factual errors or misrepresentations?
    4. Does the post generally follow the key principles of the writing guide?

    If there are only minor issues that don't significantly impact the post's quality, consider the draft acceptable.

    Current draft: {current_draft}
    Style guide: {styleguide}

    Your output should be a json object in the following format, but should **not** include the triple backticks or the 'json' label:
    {{
        "Draft Acceptable": (Yes/No),
        "Comments on current draft": (comments)
    }}
    """
)