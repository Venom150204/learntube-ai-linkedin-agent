from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
import logging
from dotenv import load_dotenv

from src.models.schemas import GraphState
from src.nodes.nodes import (
    profile_analyzer_node,
    search_node,
    job_fit_analyst_node,
    content_enhancer_node,
    career_counselor_node,
    planner_node,
    plan_executor_node
)

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


# -- Routing Functions --

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
    """Create workflow with comprehensive error handling"""
    try:
        workflow = StateGraph(GraphState)

        # Add all nodes with error handling
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

        compiled_workflow = workflow.compile(checkpointer=checkpointer)
        logger.info("Workflow compiled successfully")
        return compiled_workflow
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise RuntimeError(f"Failed to initialize workflow: {str(e)}")