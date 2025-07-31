"""
LLM utilities and retry logic for LearnTube.ai
"""
import logging
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.error_handling import MAX_RETRIES, DEFAULT_TIMEOUT

# Configure logging
logger = logging.getLogger(__name__)


def get_llm_with_fallback(model: str = "gpt-4-turbo", temperature: float = 0.7) -> ChatOpenAI:
    """
    Get LLM instance with fallback to cheaper model
    
    Args:
        model: Primary model to use
        temperature: Temperature setting
        
    Returns:
        ChatOpenAI instance
    """
    try:
        return ChatOpenAI(
            model=model, 
            temperature=temperature, 
            request_timeout=DEFAULT_TIMEOUT,
            max_retries=2
        )
    except Exception as e:
        logger.warning(f"Failed to initialize {model}, falling back to gpt-3.5-turbo: {e}")
        return ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=temperature, 
            request_timeout=DEFAULT_TIMEOUT,
            max_retries=2
        )


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def call_llm_with_retry(llm: ChatOpenAI, prompt: str) -> Any:
    """
    Call LLM with retry logic
    
    Args:
        llm: LLM instance
        prompt: Prompt to send
        
    Returns:
        LLM response
    """
    return llm.invoke([HumanMessage(content=prompt)])


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def call_structured_llm_with_retry(llm: Any, prompt: str) -> Any:
    """
    Call structured LLM with retry logic
    
    Args:
        llm: Structured LLM instance
        prompt: Prompt to send
        
    Returns:
        Structured LLM response
    """
    return llm.invoke(prompt)


def get_safe_llm_response(llm: ChatOpenAI, prompt: str, fallback_response: str = None) -> str:
    """
    Get LLM response with safe error handling
    
    Args:
        llm: LLM instance
        prompt: Prompt to send
        fallback_response: Response to use if LLM fails
        
    Returns:
        LLM response or fallback
    """
    try:
        response = call_llm_with_retry(llm, prompt)
        return response.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        if fallback_response:
            return fallback_response
        return "I'm having trouble generating a response right now. Please try again."


def truncate_content(content: str, max_length: int) -> str:
    """
    Safely truncate content to fit within token limits
    
    Args:
        content: Content to truncate
        max_length: Maximum length
        
    Returns:
        Truncated content
    """
    if len(content) <= max_length:
        return content
    
    # Try to truncate at sentence boundaries
    truncated = content[:max_length]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    
    # Use the last sentence or paragraph boundary
    boundary = max(last_period, last_newline)
    if boundary > max_length * 0.8:  # If boundary is reasonably close to end
        return content[:boundary + 1]
    
    return content[:max_length] + "..."