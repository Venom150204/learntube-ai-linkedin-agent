"""
Pydantic models and schemas for LearnTube.ai
"""
from typing import TypedDict, Annotated, List, Literal, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


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
    job_title: Optional[str] = Field(
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
class JobDescriptionRecord(TypedDict):
    """Record for each job description analyzed"""

    id: str
    title: str
    content: str
    source: Literal["pasted", "researched", "cached"]
    timestamp: str
    analysis_complete: bool


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """

    messages: Annotated[list, lambda x, y: x + y]
    profile_data: str
    job_description: Optional[str]  # Current JD being processed
    job_descriptions_history: List[JobDescriptionRecord]  # All JDs ever analyzed
    current_job_context: List[str]  # IDs of JDs currently being discussed
    researched_jobs: Dict[str, str]  # Cache for searched roles
    plan: List[str]
    original_user_request: str | None
    conversation_context: Dict[str, Any]  # Flexible context storage
    planner_decision: Dict[str, Any]  # Pass planner's intent to other nodes