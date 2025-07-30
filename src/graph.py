from typing import TypedDict, Annotated, List, Literal, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, Field
import os
import json
from dotenv import load_dotenv

load_dotenv()


# -- Pydantic Models for Structured Output & Planning --
class JobSuggestion(BaseModel):
    """A single job suggestion tailored to a user's profile."""

    job_title: str = Field(description="The suggested job title.")
    justification: str = Field(
        description="A brief, 1-2 sentence explanation of why this role is a good fit, written in the second person (e.g., 'Your experience...')."
    )


class JobSuggestions(BaseModel):
    """A list of job suggestions tailored to the user's profile."""

    suggestions: List[JobSuggestion]


class KeywordAnalysis(BaseModel):
    """An analysis of a single keyword's presence."""

    keyword: str = Field(description="A crucial keyword from the job description.")
    is_present: bool = Field(
        description="Whether the keyword is present in the user's profile."
    )
    reasoning: str = Field(
        description="A brief explanation for why the keyword is considered present or absent."
    )


class JobFitAnalysis(BaseModel):
    """A detailed analysis comparing a user's profile to a job description."""

    match_score: int = Field(
        description="A percentage from 0 to 100 representing the profile's match with the job.",
        ge=0,
        le=100,
    )
    score_reasoning: str = Field(
        description="The reasoning for the match score, based on skills and experience overlap."
    )
    keyword_analysis: List[KeywordAnalysis] = Field(
        description="An analysis of crucial keywords."
    )
    strengths: List[str] = Field(
        description="A list of reasons why the user is a strong candidate for the role."
    )
    gaps: List[str] = Field(
        description="A list of actionable suggestions to address gaps in the user's profile."
    )


class SearchIntent(BaseModel):
    """The user's intent, used for internal routing within the research node."""

    intent: Literal["suggest_roles", "analyze_specific_role"] = Field(
        description="The user's primary goal: 'suggest_roles' for general advice, or 'analyze_specific_role' for a specific title."
    )
    job_title: str | None = Field(
        description="The specific job title if the user's intent is 'analyze_specific_role'."
    )


class Plan(BaseModel):
    """A plan of sequential tasks to be executed by the agent."""

    tasks: List[
        Literal["search", "job_fit_analyst", "content_enhancer", "career_counselor"]
    ] = Field(
        description="A list of tools representing the ordered sequence of tasks to execute. The first task in the list is the first one to be executed."
    )


# -- State Definition --
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """

    messages: Annotated[list, lambda x, y: x + y]
    profile_data: str
    job_description: str | None = None
    researched_jobs: Dict[str, str] = {}
    plan: List[str] = []
    original_user_request: str | None = None


# -- Agent Nodes --


def profile_analyzer_node(state: GraphState):
    """
    Analyzes the user's profile on the first turn.
    """
    print("\n--- Running Profile Analyzer ---")
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)
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
    return {"messages": [response], "plan": []}


def _perform_search_and_synthesis(job_title: str) -> str | None:
    """Helper function to search and synthesize a job description."""
    print(f"--- Conducting research for: {job_title} ---")
    search_queries = [
        f'responsibilities and skills for a "{job_title}" role',
        f'"day in the life of a {job_title}"',
        f'example job posting for "{job_title}"',
    ]

    wrapper = SearxSearchWrapper(searx_host=os.getenv("SEARXNG_URL"))
    all_results = []
    for query in search_queries:
        print(f"  - Querying: {query}")
        try:
            results = wrapper.results(query, num_results=3, categories=["general"])
            all_results.extend(results)
        except Exception as e:
            print(f"    - Search failed for query '{query}': {e}")
    if not all_results:
        return None

    synthesizer_llm = ChatOpenAI(model="gpt-4-turbo")
    results_str = "\n".join(
        [
            f"Title: {res.get('title', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}"
            for res in all_results
        ]
    )
    synthesizer_prompt = f"""
    You are an expert research analyst. Synthesize the provided search results into a single, comprehensive job description for the role of '{job_title}'.
    Format it as a professional job description with sections for Summary, Key Responsibilities, and Required Skills.
    Base your synthesis SOLELY on the provided search results.
    Search results:
    ---
    {results_str}
    ---
    """
    print("--- Synthesizing a unified job description... ---")
    return synthesizer_llm.invoke(synthesizer_prompt).content


def search_node(state: GraphState):
    """
    A dedicated research node.
    """
    print("\n--- Running Search Node ---")
    user_request = state["messages"][-1].content
    profile_data = state.get("profile_data")

    intent_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(SearchIntent)
    intent_prompt = f"""
    Analyze the user's request to determine their search-related intent.

    1. If the user asks for suggestions or what roles would suit them, the intent is 'suggest_roles'.
    2. If the user provides a specific job title to be analyzed or compared against, the intent is 'analyze_specific_role'.

    User Request: "{user_request}"
    """
    intent_result = intent_llm.invoke(intent_prompt)
    print(f"Search Intent: {intent_result.intent}")

    if intent_result.intent == "suggest_roles":
        suggester_llm = ChatOpenAI(model="gpt-4-turbo").with_structured_output(
            JobSuggestions, method="function_calling"
        )
        suggester_prompt = f"Analyze the user's profile and suggest three relevant job titles.\n\nUser Profile: --- {profile_data} ---"
        suggestions = suggester_llm.invoke(suggester_prompt).suggestions

        researched_jobs = {}
        for suggestion in suggestions:
            jd = _perform_search_and_synthesis(suggestion.job_title)
            if jd:
                researched_jobs[suggestion.job_title] = jd

        response_text = "Based on your profile, I've researched a few roles that seem like a great fit:\n\n" + "\n".join(
            f"- **{s.job_title}**: {s.justification}" for s in suggestions
        )
        response_text += "\n\nWould you like a detailed analysis of your profile against any of these roles?"

        return {
            "messages": [AIMessage(content=response_text)],
            "researched_jobs": researched_jobs,
            "job_description": None,
        }

    elif intent_result.intent == "analyze_specific_role":
        job_title = intent_result.job_title
        if not job_title:
            return {"messages": [AIMessage(content="I'm sorry, I couldn't identify a specific job title in your request. Could you please clarify?")]}

        job_description = _perform_search_and_synthesis(job_title)
        if not job_description:
            return {"messages": [AIMessage(content=f"I'm sorry, I couldn't find enough information for '{job_title}'.")]}

        summary_message = AIMessage(
            content=f"I've researched the role of **{job_title}**. I will now proceed with the analysis."
        )
        return {"messages": [summary_message], "job_description": job_description}

    return {"messages": [AIMessage(content="I'm not sure how to handle that request. Could you rephrase?")]}


def job_fit_analyst_node(state: GraphState):
    """
    Compares the user's profile to a job description.
    """
    print("\n--- Running Job Fit Analyst ---")
    structured_llm = ChatOpenAI(model="gpt-4-turbo").with_structured_output(JobFitAnalysis, method="function_calling")

    prompt = f"""
    Provide a detailed, actionable comparison between the provided LinkedIn profile and the target job description.
    Your analysis must be completely objective and based *only* on the text provided.

    **LinkedIn Profile:**
    ---
    {state['profile_data']}
    ---
    **Target Job Description:**
    ---
    {state['job_description']}
    ---
    """
    analysis = structured_llm.invoke(prompt)

    markdown_output = f"### Job Fit Analysis\n\n"
    markdown_output += (
        f"**Overall Match Score**: **{analysis.match_score}%**\n"
        f"_{analysis.score_reasoning}_\n\n"
    )
    markdown_output += "**Keyword Analysis**:\n"
    for keyword in analysis.keyword_analysis:
        status = "✅ Present" if keyword.is_present else "❌ Missing"
        markdown_output += f"- **{keyword.keyword}**: {status}\n"
    markdown_output += "\n**Your Strengths for this Role**:\n"
    for strength in analysis.strengths:
        markdown_output += f"- {strength}\n"
    markdown_output += "\n**Actionable Gaps & Suggestions**:\n"
    for gap in analysis.gaps:
        markdown_output += f"- {gap}\n"

    return {"messages": [AIMessage(content=markdown_output)]}


def content_enhancer_node(state: GraphState):
    """
    Rewrites a specific section of the user's profile.
    """
    print("\n--- Running Proactive Content Enhancer ---")
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)
    user_request = state["original_user_request"] or state["messages"][-1].content

    sections_to_rewrite = []
    if "headline" in user_request.lower():
        sections_to_rewrite.append("headline")
    if "about" in user_request.lower():
        sections_to_rewrite.append("about")
    if "experience" in user_request.lower():
        sections_to_rewrite.append("experience")

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
        *   **For 'experience':** For my most recent job role, rewrite the description using 3-4 bullet points. Each bullet point **must be a single, flowing sentence that internally follows the STAR method (Situation, Task, Action, Result)** but **DO NOT** explicitly write the words "Situation:", "Task:", "Action:", or "Result:". For example, instead of "Situation: Faced with X... Task: Needed to Y... Action: I did Z... Result: Achieved A...", write "- Achieved A by implementing Z to solve the challenge of X, which was required for Y." Include quantifiable metrics.

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
    """
    Provides general career advice.
    """
    print("\n--- Running Career Counselor ---")
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)
    prompt = f"""
    You are a pragmatic, senior-level career coach. Your advice must be realistic and based on tangible steps.
    Use the user's profile and conversation history to inform your response, but **do not mention that you are reviewing them.** Respond directly to the user's query.

    **Your Task:**
    1.  First, determine if the user's stated or implied career goal is a significant leap from their current role (e.g., a junior developer asking to become a CTO).
    2.  **If it is a leap:** Acknowledge their ambition positively, but then ground the conversation in reality. Outline a realistic, step-by-step career ladder they would need to climb. Focus ALL of your advice on achieving the **very next step** on that ladder.
    3.  **If the goal is a reasonable next step:** Provide a targeted skill gap analysis, suggest 2-3 specific, high-impact learning resources, and offer strategic advice on networking and project selection for that role.
    
    Here is the necessary context:
    **User Profile:** --- {state['profile_data']} ---
    **Conversation History:** --- {state['messages']} ---
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


# -- Planning and Control Flow --

def planner_node(state: GraphState):
    """
    The master planner. It creates a multi-step plan based on the user's request.
    This is the primary entry point for all multi-turn conversations.
    """
    print("\n--- Running Master Planner ---")
    user_request = state["messages"][-1].content
    profile_data = state["profile_data"]
    
    if len(user_request) > 300 and "responsibilities" in user_request.lower():
        print("Planner: Pasted job description detected. Planning direct analysis.")
        tasks = ["job_fit_analyst"]
        # Check if a rewrite is also requested.
        if "rewrite" in user_request.lower() or "enhance" in user_request.lower():
            tasks.append("content_enhancer")
        return {"plan": tasks, "job_description": user_request, "original_user_request": user_request}

    planner_llm = ChatOpenAI(model="gpt-4-turbo").with_structured_output(Plan, method="function_calling")
    tools_string = """
        - `search`: Use for finding, suggesting, or researching specific job roles.
        - `job_fit_analyst`: Use for comparing the user's profile against a job description. This should always follow 'search' unless a JD is pasted.
        - `content_enhancer`: Use for rewriting or improving a profile section.
        - `career_counselor`: Use for general career advice.
    """
    prompt = f"""
    You are an expert AI task planner. Create a logical, step-by-step plan to address the user's request using the available tools.

    **User's Profile Summary:**
    ---
    {profile_data[:300]}...
    ---

    **User's Request:**
    ---
    "{user_request}"
    ---

    **Available Tools:**
    {tools_string}

    Create a plan as a list of tool names in the precise order of execution.
    """
    plan_result = planner_llm.invoke(prompt)
    tasks = plan_result.tasks
    print(f"Planner: Generated plan: {tasks}")

    if "job_fit_analyst" in tasks and state.get("researched_jobs"):
        for job_title, jd in state["researched_jobs"].items():
            if job_title.lower() in user_request.lower():
                print(f"Planner: Found cached job '{job_title}'. Modifying plan.")
                if 'search' in tasks:
                    tasks.remove('search')
                return {"plan": tasks, "job_description": jd, "original_user_request": user_request}

    return {"plan": tasks, "original_user_request": user_request}


def plan_executor_node(state: GraphState):
    """
    The central control unit. Updates the plan and routes to the next task.
    """
    print("\n--- Executing Plan ---")
    plan = state.get("plan", [])
    if not plan:
        print("Execution Complete. Ending.")
        return {"plan": []}  # Return empty plan to signal completion
    
    # Remove the first task from the plan
    remaining_plan = plan[1:]
    print(f"Remaining tasks after current: {remaining_plan}")
    return {"plan": remaining_plan}

def route_next_task(state: GraphState):
    """
    Conditional edge function to determine the next task or end.
    """
    plan = state.get("plan", [])
    if not plan:
        return END
    
    next_task = plan[0]
    print(f"Routing to next task: {next_task}")
    return next_task


def create_workflow(checkpointer: BaseCheckpointSaver):
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("profile_analyzer", profile_analyzer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("search", search_node)
    workflow.add_node("job_fit_analyst", job_fit_analyst_node)
    workflow.add_node("content_enhancer", content_enhancer_node)
    workflow.add_node("career_counselor", career_counselor_node)
    workflow.add_node("plan_executor", plan_executor_node)

    # Define the entry point
    def route_to_planner(state: GraphState):
        return "profile_analyzer" if len(state["messages"]) <= 1 else "planner"

    workflow.set_conditional_entry_point(
        route_to_planner,
        {
            "profile_analyzer": "profile_analyzer",
            "planner": "planner",
        },
    )

    # Define the main operational loop
    workflow.add_conditional_edges(
        "planner",
        route_next_task,
        {
            "search": "search",
            "job_fit_analyst": "job_fit_analyst", 
            "content_enhancer": "content_enhancer",
            "career_counselor": "career_counselor",
            END: END,
        },
    )
    
    # After a task, go to plan executor to update plan, then route to next task
    workflow.add_edge("search", "plan_executor")
    workflow.add_edge("job_fit_analyst", "plan_executor")
    workflow.add_edge("content_enhancer", "plan_executor")
    workflow.add_edge("career_counselor", "plan_executor")
    
    # The initial profile analysis is a one-off task
    workflow.add_edge("profile_analyzer", END)

    # The executor updates the plan and routes to the next task
    workflow.add_conditional_edges(
        "plan_executor",
        route_next_task,
        {
            "search": "search",
            "job_fit_analyst": "job_fit_analyst",
            "content_enhancer": "content_enhancer", 
            "career_counselor": "career_counselor",
            END: END,
        },
    )

    return workflow.compile(checkpointer=checkpointer)
