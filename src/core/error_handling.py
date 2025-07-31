"""
Error handling utilities and decorators for LearnTube.ai
"""

import logging
import time
import traceback
from functools import wraps
from typing import Dict, Any

from langchain_core.messages import AIMessage
from src.models.schemas import GraphState

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
MAX_PROFILE_LENGTH = 50000
MAX_MESSAGE_LENGTH = 10000
MAX_CONTEXT_TOKENS = 100000


def validate_state_inputs(state: GraphState) -> Dict[str, Any]:
    """
    Validate and sanitize state inputs

    Args:
        state: Current graph state

    Returns:
        Dict with validated inputs
    """
    validated = {}

    # Validate profile data
    profile_data = state.get("profile_data", "")
    if profile_data and len(profile_data) > MAX_PROFILE_LENGTH:
        logger.warning(
            f"Profile data truncated from {len(profile_data)} to {MAX_PROFILE_LENGTH}"
        )
        profile_data = profile_data[:MAX_PROFILE_LENGTH]
    validated["profile_data"] = profile_data

    # Validate messages
    messages = state.get("messages", [])
    if messages:
        try:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                content = last_message.content
                if len(content) > MAX_MESSAGE_LENGTH:
                    content = content[:MAX_MESSAGE_LENGTH]
                validated["user_request"] = content
            else:
                validated["user_request"] = str(last_message)[:MAX_MESSAGE_LENGTH]
        except (IndexError, AttributeError):
            validated["user_request"] = ""
    else:
        validated["user_request"] = ""

    # Validate other fields
    validated["job_history"] = state.get("job_descriptions_history", [])
    validated["job_description"] = state.get("job_description")
    validated["original_request"] = state.get("original_user_request", "")

    return validated


def handle_llm_error(error: Exception, node_name: str) -> str:
    """
    Generate user-friendly error messages based on error type

    Args:
        error: The exception that occurred
        node_name: Name of the node where error occurred

    Returns:
        User-friendly error message
    """
    error_str = str(error).lower()

    if "token" in error_str or "context" in error_str:
        return (
            "The conversation has become too long. Please start a new analysis session."
        )
    elif "api" in error_str or "openai" in error_str:
        return (
            "AI service is temporarily unavailable. Please try again in a few moments."
        )
    elif "rate limit" in error_str:
        return "We're experiencing high demand. Please try again in a moment."
    elif "timeout" in error_str:
        return "The request took too long to process. Please try with a simpler query."
    elif "validation" in error_str:
        return "There was an issue with the data format. Please try rephrasing your request."
    else:
        return f"I encountered a technical issue while {node_name.replace('_', ' ')}. Please try again or rephrase your request."


class NodeError(Exception):
    """Custom exception for node execution errors"""

    pass


class ValidationError(Exception):
    """Custom exception for input validation errors"""

    pass
