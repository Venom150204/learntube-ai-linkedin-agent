import os
import json
import re
import time
import logging
from typing import Optional, Dict, Any
from apify_client import ApifyClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_TIMEOUT = 120
MAX_RETRIES = 3
MIN_PROFILE_LENGTH = 100

class LinkedInScrapingError(Exception):
    """Custom exception for LinkedIn scraping errors"""
    pass

def validate_linkedin_url(profile_url: str) -> str:
    """
    Validate and extract username from LinkedIn URL
    
    Args:
        profile_url: LinkedIn profile URL
        
    Returns:
        str: Extracted username
        
    Raises:
        LinkedInScrapingError: If URL is invalid
    """
    if not profile_url or not isinstance(profile_url, str):
        raise LinkedInScrapingError("Profile URL must be a non-empty string")
    
    # Normalize URL
    profile_url = profile_url.strip()
    
    # Add https if missing
    if not profile_url.startswith(('http://', 'https://')):
        profile_url = 'https://' + profile_url
    
    # Extract username with comprehensive regex
    patterns = [
        r"linkedin\.com/in/([^/?&]+)",  # Standard format
        r"linkedin\.com/pub/([^/?&]+)",  # Public format
        r"linkedin\.com/profile/view\?id=([^/?&]+)",  # Old format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, profile_url, re.IGNORECASE)
        if match:
            username = match.group(1)
            # Clean username
            username = username.strip('/')
            if username:
                logger.info(f"Extracted username: {username}")
                return username
    
    raise LinkedInScrapingError(
        "Invalid LinkedIn Profile URL format. "
        "Expected formats: linkedin.com/in/username, linkedin.com/pub/username"
    )

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def call_apify_actor(client: ApifyClient, username: str) -> Dict[str, Any]:
    """
    Call Apify actor with retry logic
    
    Args:
        client: Apify client instance
        username: LinkedIn username
        
    Returns:
        Dict containing actor run results
        
    Raises:
        LinkedInScrapingError: If actor call fails
    """
    run_input = {"username": username}
    
    logger.info(f"Calling Apify actor for username: {username}")
    
    try:
        run = client.actor("apimaestro/linkedin-profile-detail").call(
            run_input=run_input,
            timeout_secs=DEFAULT_TIMEOUT
        )
        
        if not run or "defaultDatasetId" not in run:
            raise LinkedInScrapingError("Apify actor returned invalid response")
        
        return run
        
    except Exception as e:
        logger.error(f"Apify actor call failed: {e}")
        if "rate limit" in str(e).lower():
            raise LinkedInScrapingError("Rate limit exceeded. Please try again later.")
        elif "timeout" in str(e).lower():
            raise LinkedInScrapingError("Request timed out. The profile may be too large or service is busy.")
        elif "unauthorized" in str(e).lower():
            raise LinkedInScrapingError("Invalid API key or unauthorized access.")
        else:
            raise LinkedInScrapingError(f"Actor execution failed: {str(e)}")

def validate_scraped_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean scraped profile data
    
    Args:
        data: Raw scraped data
        
    Returns:
        Dict: Validated and cleaned data
        
    Raises:
        LinkedInScrapingError: If data is invalid
    """
    if not data:
        raise LinkedInScrapingError("No profile data returned")
    
    # Check for error in scraped data
    if "error" in data:
        raise LinkedInScrapingError(f"Scraping error: {data['error']}")
    
    # Validate essential fields
    essential_fields = ["fullName", "headline", "summary"]
    missing_fields = [field for field in essential_fields if not data.get(field)]
    
    if len(missing_fields) >= 2:  # Allow one missing field
        logger.warning(f"Profile missing essential fields: {missing_fields}")
    
    # Check minimum profile completeness
    profile_text = json.dumps(data, indent=2)
    if len(profile_text) < MIN_PROFILE_LENGTH:
        raise LinkedInScrapingError("Profile data appears incomplete or empty")
    
    # Clean sensitive information
    sensitive_fields = ["email", "phone", "address"]
    for field in sensitive_fields:
        if field in data:
            data[field] = "[REDACTED]"
    
    logger.info(f"Profile validation passed - {len(profile_text)} characters")
    return data

def scrape_linkedin_profile(profile_url: str) -> str:
    """
    Scrapes a LinkedIn profile using the Apify actor with comprehensive error handling.
    
    Args:
        profile_url: LinkedIn profile URL
        
    Returns:
        str: JSON string of scraped profile data
        
    Raises:
        LinkedInScrapingError: If scraping fails
    """
    start_time = time.time()
    
    try:
        # Validate API key
        apify_api_key = os.getenv("APIFY_API_KEY")
        if not apify_api_key:
            raise LinkedInScrapingError(
                "APIFY_API_KEY environment variable is not set. "
                "Please configure your Apify API key."
            )
        
        # Validate and extract username
        username = validate_linkedin_url(profile_url)
        
        # Initialize Apify client
        try:
            client = ApifyClient(apify_api_key, timeout_secs=DEFAULT_TIMEOUT)
        except Exception as e:
            raise LinkedInScrapingError(f"Failed to initialize Apify client: {str(e)}")
        
        # Call actor with retry logic
        run = call_apify_actor(client, username)
        
        # Fetch results
        try:
            logger.info("Fetching scraped data from dataset...")
            dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
        except Exception as e:
            raise LinkedInScrapingError(f"Failed to fetch results: {str(e)}")
        
        # Validate results
        if not dataset_items:
            raise LinkedInScrapingError(
                "No profile data returned. The profile may be private, "
                "deleted, or the username might be incorrect."
            )
        
        # Validate and clean data
        profile_data = validate_scraped_data(dataset_items[0])
        
        # Convert to JSON
        profile_json = json.dumps(profile_data, indent=2, ensure_ascii=False)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Profile scraping completed successfully in {elapsed_time:.2f}s")
        
        return profile_json
        
    except LinkedInScrapingError:
        # Re-raise our custom exceptions
        raise
    except json.JSONEncodeError as e:
        logger.error(f"JSON encoding error: {e}")
        raise LinkedInScrapingError("Failed to encode profile data as JSON")
    except Exception as e:
        logger.error(f"Unexpected scraping error: {e}")
        raise LinkedInScrapingError(
            f"An unexpected error occurred while scraping the profile: {str(e)}"
        )

def get_profile_summary(profile_data: str) -> Dict[str, Any]:
    """
    Extract key information from scraped profile for validation
    
    Args:
        profile_data: JSON string of profile data
        
    Returns:
        Dict with summary information
    """
    try:
        data = json.loads(profile_data)
        
        summary = {
            "name": data.get("fullName", "Unknown"),
            "headline": data.get("headline", "No headline"),
            "location": data.get("location", {}).get("country", "Unknown"),
            "connections": data.get("connectionsCount", 0),
            "experience_count": len(data.get("positions", [])),
            "education_count": len(data.get("schools", [])),
            "skills_count": len(data.get("skills", [])),
            "data_size": len(profile_data)
        }
        
        return summary
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON data"}
    except Exception as e:
        return {"error": f"Failed to parse profile: {str(e)}"}

# Health check function for monitoring
def test_scraping_service() -> Dict[str, Any]:
    """
    Test the scraping service health
    
    Returns:
        Dict with service status
    """
    try:
        apify_api_key = os.getenv("APIFY_API_KEY")
        if not apify_api_key:
            return {
                "status": "unhealthy",
                "error": "APIFY_API_KEY not configured"
            }
        
        # Try to initialize client
        client = ApifyClient(apify_api_key, timeout_secs=10)
        
        # Test connection (this will validate the API key)
        try:
            # Get user info to validate API key
            user_info = client.user().get()
            return {
                "status": "healthy",
                "api_key_valid": True,
                "user_id": user_info.get("id", "unknown")
            }
        except Exception as e:
            return {
                "status": "degraded",
                "api_key_valid": False,
                "error": str(e)
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": f"Service check failed: {str(e)}"
        }