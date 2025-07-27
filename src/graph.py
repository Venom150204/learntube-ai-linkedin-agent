from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langgraph.checkpoint.base import BaseCheckpointSaver
import re
import json

# -- State Definition --
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    profile_data: str
    job_description: str = None
    next_node: str = None

# -- Agent Nodes (Upgraded with Intelligence & Logging) --

def profile_analyzer_node(state: GraphState):
    print("\n--- Running Profile Analyzer ---")
    llm = ChatOpenAI(model="gpt-4o")
    try:
        profile_data = json.loads(state.get('profile_data', '{}'))
        if 'error' in profile_data:
            error_message = f"I'm sorry, an error occurred while scraping the profile: {profile_data['error']}"
            return {"messages": [AIMessage(content=error_message)]}
    except (json.JSONDecodeError, TypeError):
        error_message = "I'm sorry, but the scraped data was not in a valid format. Please try again."
        return {"messages": [AIMessage(content=error_message)]}
    prompt = f"""
    You are a world-class LinkedIn profile optimization expert, speaking in the first person. Your analysis should be structured, insightful, and professional.
    Analyze the following LinkedIn profile and provide a summary of strengths and weaknesses.
    **Structure your response as follows:**
    1.  **Overall Impression:** Start with a brief, one-sentence summary of the profile.
    2.  **Key Strengths:** Provide 3-5 bullet points highlighting what the user is doing well.
    3.  **Critical Areas for Improvement:** Provide 3-5 bullet points on the most important things to fix.
    4.  **Next Steps:** Conclude by encouraging me to ask follow-up questions, such as "You can ask me to rewrite your 'About' section, or we can analyze your profile against a specific job role."
    Profile Data: {state['profile_data']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

def search_node(state: GraphState):
    print("\n--- Running Search Node ---")
    user_request = state["messages"][-1].content
    job_title = ""

    # 1. First, try a simple, fast regex to find a job title in quotes
    match = re.search(r'["\'](.*?)["\']', user_request)
    if match:
        job_title = match.group(1)
        print(f"Extracted Job Title via Regex: {job_title}")

    # 2. If regex fails, use the LLM as a fallback for conversational queries
    if not job_title:
        print("Regex found no title, falling back to LLM extractor...")
        extractor_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        extractor_prompt = f"""
        Extract the most logical and concise job title from the following user request.
        For example, if the user says "jobs that are focused on Gen AI more", you should extract "Generative AI Engineer".
        If the user says "I want to be a product manager", you should extract "Product Manager".
        Only return the job title and nothing else.
        User Request: "{user_request}"
        """
        job_title = extractor_llm.invoke(extractor_prompt).content.strip()
        print(f"Extracted Job Title via LLM: {job_title}")
    
    search_tool = DuckDuckGoSearchRun()
    search_query = f"detailed job description for a {job_title} including key skills and responsibilities"
    print(f"Performing search for: {search_query}")
    
    job_description = search_tool.invoke(search_query)
    
    ai_message = AIMessage(content=f"I've found a job description for a '{job_title}'. Now, I will analyze how your profile matches up.")
    return {"job_description": job_description, "messages": [ai_message]}

def job_fit_analyst_node(state: GraphState):
    print("\n--- Running Job Fit Analyst ---")
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"""
    You are an expert job fit analyst, speaking in the first person. Your task is to provide a detailed, actionable comparison between the user's profile and a target job description.

    **My LinkedIn Profile:** --- {state['profile_data']} ---
    **Target Job Description:** --- {state['job_description']} ---

    **Your analysis must include:**
    1.  **Overall Match Score:** A percentage from 0% to 100%. Explain your reasoning for this score based on the overlap of key skills and years of experience.
    2.  **Keyword Analysis:** Identify 5-7 crucial keywords from the job description and state whether they are present in my profile.
    3.  **My Strengths:** 3-5 bullet points explaining why I am a strong candidate, referencing specific parts of my profile.
    4.  **Actionable Gaps & Suggestions:** 3-5 bullet points identifying what is missing from my profile. For each point, provide a concrete suggestion for what I should add or change.
    
    If the job description seems completely irrelevant to my profile, state that you couldn't find a good match and ask me to provide a more specific job title.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

def content_enhancer_node(state: GraphState):
    print("\n--- Running Proactive Content Enhancer ---")
    llm = ChatOpenAI(model="gpt-4o")
    user_request = state["messages"][-1].content

    # New: Identify all sections mentioned by the user
    sections_to_rewrite = []
    if "headline" in user_request.lower():
        sections_to_rewrite.append("headline")
    if "about" in user_request.lower():
        sections_to_rewrite.append("about")
    if "experience" in user_request.lower():
        sections_to_rewrite.append("experience")
    
    # If no specific section is mentioned, default to a full review
    if not sections_to_rewrite:
        sections_to_rewrite = ["headline", "about", "experience"]
        print(f"No specific section found in request, defaulting to full review of: {sections_to_rewrite}")
    else:
        print(f"Identified sections for rewrite: {sections_to_rewrite}")

    prompt = f"""
    You are an expert LinkedIn copywriter and career strategist, speaking in the first person. Your tone is professional, confident, and results-oriented.

    I want you to enhance my LinkedIn profile based on my request. Here is your task:

    1.  **Generate Rewrites:** For each of the following sections I've asked for (`{', '.join(sections_to_rewrite)}`), provide a rewritten, enhanced version based on the rules below. Present each rewritten section under a clear markdown heading (e.g., `### Enhanced About Section`).
        *   **For 'headline':** Create a concise, keyword-rich headline (under 220 characters) that includes my current role and key area of expertise (e.g., "Software Engineer at XYZ | Building with Generative AI & LLMs").
        *   **For 'about':** Write a compelling, first-person summary (under 150 words) that tells a story about my passion and impact.
        *   **For 'experience':** For my most recent job role, rewrite the description using 3-4 bullet points. Each bullet point must follow the STAR method (Situation, Task, Action, Result) and include quantifiable metrics where possible (e.g., "Reduced data processing time by 30% by implementing...").

    2.  **Be Proactive:** After you have rewritten the sections I asked for, analyze my entire profile. Identify the **single most valuable section** that I *did not* ask you to improve. Then, provide an enhanced version of it under the heading `### Proactive Suggestion:`.

    **My Full Profile:**
    ---
    {state['profile_data']}
    ---

    **My Original Request:**
    ---
    "{user_request}"
    ---

    Provide only the rewritten sections as your response. Do not add any extra conversational text or commentary.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

def career_counselor_node(state: GraphState):
    print("\n--- Running Career Counselor ---")
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"""
    You are a pragmatic, senior-level career coach, speaking in the first person. Your advice must be realistic and based on tangible steps.
    Analyze my profile and our conversation history to understand my current level and goals.
    **My Profile:** --- {state['profile_data']} ---
    **Our Conversation:** --- {state['messages']} ---
    **Your Task:**
    1.  First, determine if my stated or implied career goal is a significant leap from my current role (e.g., a junior developer asking to become a CTO).
    2.  **If it is a leap:** Acknowledge my ambition positively, but then ground our conversation in reality. Outline a realistic, step-by-step career ladder I would need to climb. Then, focus ALL of your advice on achieving the **very next step** on that ladder.
    3.  **If my goal is a reasonable next step:** Provide a targeted skill gap analysis, suggest 2-3 specific, high-impact learning resources, and offer strategic advice on networking and project selection for that role.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# -- Conditional Logic (Definitive Architecture) --

def router_node(state: GraphState):
    print("\n--- Running Master Router ---")
    user_request = state["messages"][-1].content
    
    # New: Check if the user has provided a job description in their message
    jd_checker_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    jd_checker_prompt = f"""
    Does the following user message contain a job description for analysis? 
    Answer with a single word: 'yes' or 'no'.
    User message: "{user_request}"
    """
    contains_jd = jd_checker_llm.invoke(jd_checker_prompt).content.strip().lower()
    print(f"Router check: Does the message contain a job description? --> {contains_jd}")

    if "yes" in contains_jd:
        # If a JD is provided, extract it and go directly to the analyst
        print("Routing decision: JD found in message, extracting and routing directly to Job Fit Analyst.")
        return {"job_description": user_request, "next_node": "job_fit_analyst"}

    # On the first turn (without a JD), always go to the analyzer
    if len(state['messages']) <= 1:
        print("Routing decision: First turn, directing to Profile Analyzer.")
        return {"next_node": "profile_analyzer"}

    # On subsequent turns without a JD, use the LLM to choose the next agent
    print("Routing decision: No JD in message, using LLM to choose next agent.")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [
        {"type": "function", "function": {"name": "job_fit_analyst", "description": "Used when the user wants to compare their profile to a job description but has NOT provided one."}},
        {"type": "function", "function": {"name": "content_enhancer", "description": "Used when the user wants to rewrite or improve a section of their profile."}},
        {"type": "function", "function": {"name": "career_counselor", "description": "Used for general career advice, skill gap analysis, or long-term planning."}},
    ]
    prompt = f"""
    You are an expert at routing user requests. Based on the user's last message, choose the correct specialist tool.
    **User's Last Message:** "{user_request}"
    """
    response = llm.invoke([HumanMessage(content=prompt)], tools=tools)
    
    if not response.tool_calls:
        next_node = "career_counselor"
    else:
        next_node = response.tool_calls[0]['name']
    
    print(f"Routing decision: Directing to '{next_node}'.")
    return {"next_node": next_node}

def create_workflow(checkpointer: BaseCheckpointSaver):
    workflow = StateGraph(GraphState)
    workflow.add_node("router", router_node)
    workflow.add_node("profile_analyzer", profile_analyzer_node)
    workflow.add_node("search", search_node)
    workflow.add_node("job_fit_analyst", job_fit_analyst_node)
    workflow.add_node("content_enhancer", content_enhancer_node)
    workflow.add_node("career_counselor", career_counselor_node)
    
    workflow.set_entry_point("router")

    # This single conditional edge is the brain of the operation.
    # It routes based on the decision made in the router_node.
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_node"],
        {
            "profile_analyzer": "profile_analyzer",
            "search": "search",
            "job_fit_analyst": "job_fit_analyst", # This is the direct route when a JD is provided by the user.
            "content_enhancer": "content_enhancer",
            "career_counselor": "career_counselor",
        }
    )
    
    workflow.add_edge("search", "job_fit_analyst")
    
    workflow.add_edge("profile_analyzer", END)
    workflow.add_edge("job_fit_analyst", END)
    workflow.add_edge("content_enhancer", END)
    workflow.add_edge("career_counselor", END)
    
    return workflow.compile(checkpointer=checkpointer)
