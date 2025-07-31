"""
Context Manager for intelligent conversation understanding
"""
from typing import List, Dict, Optional
import re
from datetime import datetime
import uuid

class ConversationContextManager:
    """
    Manages conversation context and understands user intent across multiple turns.
    Handles references, comparisons, and context switching intelligently.
    """
    
    @staticmethod
    def extract_job_title_from_content(content: str) -> str:
        """Extract the most likely job title from job description content"""
        # Look for common patterns
        patterns = [
            r"(?:position|role|title)[\s:]+([A-Za-z\s]+?)(?:\n|$)",
            r"(?:seeking|hiring|looking for)[\s:]+(?:a\s+)?([A-Za-z\s]+?)(?:\n|$|with)",
            r"^([A-Za-z\s]+?)(?:\n|$)",  # First line often contains title
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 3 and len(title) < 50:  # Reasonable title length
                    return title
        
        # Fallback: Look for role-related keywords
        if "data scientist" in content.lower():
            return "Data Scientist"
        elif "machine learning" in content.lower():
            return "Machine Learning Engineer"
        elif "software engineer" in content.lower():
            return "Software Engineer"
        elif "data engineer" in content.lower():
            return "Data Engineer"
        
        return "Professional Role"
    
    @staticmethod
    def understand_user_intent(
        user_request: str, 
        job_history: List[Dict],
        recent_messages: List[str]
    ) -> Dict:
        """
        Understands what the user is trying to do based on:
        - Current request
        - Job description history
        - Recent conversation context
        """
        lower_request = user_request.lower()
        
        # Check for comparison requests
        comparison_patterns = [
            r"compare.*(?:both|these|the)\s+(?:jobs?|roles?|positions?)",
            r"which.*(?:better|suits?|fits?)",
            r"(?:between|among).*(?:jobs?|roles?)",
            r"difference.*between",
            r"(?:first|second|previous).*(?:job|role).*(?:or|vs)",
        ]
        
        for pattern in comparison_patterns:
            if re.search(pattern, lower_request):
                return {
                    "type": "comparison",
                    "job_ids": [job["id"] for job in job_history[-2:]],  # Last 2 jobs
                    "details": "User wants to compare multiple job opportunities"
                }
        
        # Check for references to previous jobs
        reference_patterns = [
            (r"(?:that|the)\s+(?:previous|last|first|second)\s+(?:job|role|position)", "previous"),
            (r"(?:go back to|about)\s+(?:the|that)\s+(\w+\s*\w*)\s+(?:role|job|position)", "specific"),
            (r"(?:the|that)\s+(?:one|role|job|position)\s+(?:we|I)\s+(?:discussed|analyzed|looked at)", "previous"),
        ]
        
        for pattern, ref_type in reference_patterns:
            match = re.search(pattern, lower_request)
            if match:
                if ref_type == "specific" and match.group(1):
                    # Try to find the specific role mentioned
                    role_name = match.group(1).strip()
                    for job in reversed(job_history):
                        if role_name.lower() in job["title"].lower():
                            return {
                                "type": "reference",
                                "job_id": job["id"],
                                "details": f"User is referring to the {job['title']} role"
                            }
                elif job_history:
                    return {
                        "type": "reference",
                        "job_id": job_history[-1]["id"],
                        "details": "User is referring to a previously discussed role"
                    }
        
        # Check if it's a follow-up question about current context
        follow_up_indicators = [
            "what skills", "how can i", "should i", "what about", 
            "tell me more", "focus on", "improve", "gaps"
        ]
        
        if any(indicator in lower_request for indicator in follow_up_indicators) and job_history:
            return {
                "type": "follow_up",
                "job_ids": [job["id"] for job in job_history if job.get("analysis_complete")],
                "details": "User is asking follow-up questions about analyzed roles"
            }
        
        # Check if user pasted a job description (check this FIRST before new_analysis)
        if ConversationContextManager.looks_like_job_description(user_request):
            return {
                "type": "pasted_jd",
                "details": "User pasted a job description directly"
            }
        
        # Check if it's a new job analysis request
        new_job_indicators = [
            r"analyze.*(?:my fit|profile).*(?:for|against)",
            r"(?:how|am i).*(?:fit|qualified).*(?:for|as)",
            r"(?:role|position|job).*(?:of|as)\s+(\w+)",
        ]
        
        for pattern in new_job_indicators:
            if re.search(pattern, lower_request):
                return {
                    "type": "new_analysis",
                    "details": "User wants to analyze fit for a new role"
                }
        
        # Default to general request
        return {
            "type": "general",
            "details": "General career-related request"
        }
    
    @staticmethod
    def looks_like_job_description(text: str) -> bool:
        """Enhanced detection of whether text is a job description"""
        indicators = [
            "responsibilities", "requirements", "qualifications", "experience",
            "skills", "essential", "preferred", "minimum", "years", "yrs",
            "expertise", "knowledge", "proficiency", "must have", "nice to have",
            "job description", "we are looking", "role involves", "you will"
        ]
        
        lower_text = text.lower()
        indicator_count = sum(1 for ind in indicators if ind in lower_text)
        
        # Multiple heuristics
        return (
            (len(text) > 200 and indicator_count >= 3) or
            (len(text) > 400 and indicator_count >= 2) or
            (text.count('\n') > 5 and indicator_count >= 2) or
            ("essential" in lower_text and "preferred" in lower_text) or
            (indicator_count >= 4)
        )
    
    @staticmethod
    def create_job_record(
        content: str, 
        source: str = "pasted",
        title: Optional[str] = None
    ) -> Dict:
        """Create a standardized job description record"""
        if not title:
            title = ConversationContextManager.extract_job_title_from_content(content)
        
        return {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": content,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "analysis_complete": False
        }
    
    @staticmethod
    def get_context_summary(job_history: List[Dict]) -> str:
        """Generate a summary of the conversation context for LLMs"""
        if not job_history:
            return "No jobs have been analyzed yet."
        
        summary = f"Job history ({len(job_history)} roles analyzed):\n"
        for i, job in enumerate(job_history[-3:], 1):  # Show last 3
            status = "✓" if job.get("analysis_complete") else "..."
            summary += f"{i}. {job['title']} [{status}] - {job['source']}\n"
        
        if len(job_history) > 3:
            summary += f"... and {len(job_history) - 3} more\n"
        
        return summary