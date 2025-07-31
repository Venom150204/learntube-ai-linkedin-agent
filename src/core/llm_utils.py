"""
LLM utilities and retry logic for LearnTube.ai
"""
import logging
from typing import Any
from langchain_groq import ChatGroq
import os
from langchain_core.messages import HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.error_handling import MAX_RETRIES, DEFAULT_TIMEOUT

# Configure logging
logger = logging.getLogger(__name__)


def get_llm_with_fallback(model: str = "llama-3.3-70b-versatile", temperature: float = 0.7) -> ChatGroq:
    """
    Get LLM instance with fallback to cheaper model
    
    Args:
        model: Primary model to use
        temperature: Temperature setting
        
    Returns:
        ChatOpenAI instance
    """
    try:
        return ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_retries=2
        )
    except Exception as e:
        logger.warning(f"Failed to initialize {model}, falling back to llama-3.1-8b-instant: {e}")
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_retries=2
        )


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def call_llm_with_retry(llm: ChatGroq, prompt: str) -> Any:
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


def get_safe_llm_response(llm: ChatGroq, prompt: str, fallback_response: str = None) -> str:
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


def extract_llm_content(response: Any) -> str:
    """
    Extract text content from LLM response regardless of format
    
    Args:
        response: LLM response object
        
    Returns:
        String content from the response
    """
    # Handle different response formats
    if hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    elif hasattr(response, 'message'):
        if hasattr(response.message, 'content'):
            return response.message.content
        return str(response.message)
    elif isinstance(response, str):
        return response
    else:
        # Last resort - convert to string
        return str(response)


def clean_json_response(content: str) -> str:
    """
    Clean JSON response by removing markdown code blocks and extra content
    
    Args:
        content: Raw response content
        
    Returns:
        Cleaned JSON string
    """
    import re
    
    content = content.strip()
    
    # Remove markdown code blocks
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    
    # Find the end of JSON - look for closing brace followed by ``` or newline
    # This handles cases where LLM adds text after JSON
    if "```" in content:
        # If there's a closing markdown block, extract content before it
        content = content.split("```")[0]
    
    # Try to find complete JSON object by matching braces
    try:
        # Find the last closing brace that completes the JSON
        brace_count = 0
        json_end = -1
        
        for i, char in enumerate(content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end > 0:
            content = content[:json_end]
    except:
        pass
    
    return content.strip()


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