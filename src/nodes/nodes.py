from typing import Dict, Optional, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import SearxSearchWrapper
from pydantic import ValidationError
import os
import logging
import traceback

from src.models.schemas import (
    GraphState, 
    JobFitAnalysis,
    SearchIntent,
    Plan
)
from src.core.error_handling import (

    MAX_PROFILE_LENGTH,
    MAX_MESSAGE_LENGTH
)
from src.core.llm_utils import (
    get_llm_with_fallback,
    call_llm_with_retry,
    extract_llm_content,
    clean_json_response
)
from src.context_manager import ConversationContextManager

logger = logging.getLogger(__name__)



def profile_analyzer_node(state: GraphState):
    """
    Analyzes the user's profile on the first turn.
    """
    print("\n--- Running Profile Analyzer ---")
    
    # Validate profile data
    profile_data = state.get('profile_data', '')
    if not profile_data:
        return {
            "messages": [AIMessage(content="I couldn't find any profile data to analyze. Please make sure your LinkedIn profile was loaded correctly.")],
            "plan": []
        }
    
    # Truncate if too long
    if len(profile_data) > MAX_PROFILE_LENGTH:
        logger.warning(f"Profile data truncated from {len(profile_data)} to {MAX_PROFILE_LENGTH}")
        profile_data = profile_data[:MAX_PROFILE_LENGTH]
    
    llm = get_llm_with_fallback(model="llama-3.3-70b-versatile", temperature=0.7)
    
    prompt = f"""
    You are a world-class LinkedIn profile optimization expert, speaking in the first person. Your analysis should be structured, insightful, and professional.
    Analyze the following LinkedIn profile and provide a summary of strengths and weaknesses.
    
    IMPORTANT: Use proper markdown formatting with clear headings and bullet points.
    
    Structure your response EXACTLY as follows:
    
    ## 1. Overall Impression
    [One sentence summary]
    
    ## 2. Key Strengths
    • [Strength 1]
    • [Strength 2] 
    • [Strength 3]
    • [Add more if needed]
    
    ## 3. Critical Areas for Improvement
    • [Improvement 1]
    • [Improvement 2]
    • [Improvement 3]
    • [Add more if needed]
    
    ## 4. Next Steps
    [Encourage follow-up questions]
    
    Profile Data: {profile_data}
    """
    
    try:
        response = call_llm_with_retry(llm, prompt)
        return {"messages": [response], "plan": []}
    except Exception as e:
        logger.error(f"Failed to analyze profile after retries: {e}")
        return {
            "messages": [AIMessage(content="I'm experiencing technical difficulties analyzing your profile. Please try again later or contact support if the issue persists.")],
            "plan": []
        }


def _fallback_role_suggestions(profile_data: str, job_history: List) -> Dict:
    """Provide comprehensive role suggestions when direct search isn't needed"""
    print("Generating tailored role suggestions based on profile")

    # Use LLM for deep profile analysis and role matching
    role_advisor = get_llm_with_fallback(model="llama-3.3-70b-versatile", temperature=0.3)

    role_suggestion_prompt = f"""
    As a career advisor, analyze this profile and suggest 4-5 job roles that are PERFECT matches.
    
    Profile: {profile_data[:1200]}...
    
    For each suggested role, provide a COMPREHENSIVE overview:
    
    **[Role Title]**
    ✓ Why this role: [One sentence explaining the fit]
    
    📋 Key Responsibilities:
    • [Responsibility 1]
    • [Responsibility 2] 
    • [Responsibility 3]
    • [Responsibility 4]
    
    🛠️ Required Skills (that match your profile):
    • [Skill 1 they have]
    • [Skill 2 they have]
    • [Skill 3 they have]
    
    📈 Career Path:
    • Current role → [This suggested role] → [Next level] → [Senior level]
    
    💰 Typical Salary Range: [Range based on location and experience]
    
    🎯 Why you're a great fit: [2-3 sentences explaining specific matches between their profile and this role]
    
    ---
    
    RULES:
    1. Only suggest roles that match their CURRENT experience level
    2. Focus on roles where they already have 70%+ of required skills
    3. Include both traditional and emerging roles in their field
    4. Be specific about salary ranges and career progression
    """

    role_suggestions = role_advisor.invoke(role_suggestion_prompt).content

    # Format response
    response_text = "Based on your profile analysis, here are job roles that would be excellent matches for you:\n\n"
    response_text += role_suggestions
    response_text += "\n\n💡 **Next Steps:** Would you like me to:\n"
    response_text += "- Provide a detailed fit analysis for any of these roles?\n"
    response_text += "- Suggest specific skills to develop for your target role?\n"
    response_text += "- Help optimize your profile to attract these opportunities?"

    # Extract role titles and research them
    researched_jobs = {}
    role_lines = role_suggestions.split("\n")
    for line in role_lines:
        if line.startswith("**") and line.endswith("**"):
            role_title = line.strip("*").strip()
            if role_title:
                jd = _perform_search_and_synthesis(role_title)
                if jd:
                    researched_jobs[role_title] = jd

    # Save suggested roles to history
    updated_job_history = job_history.copy()
    for role_title, jd in researched_jobs.items():
        job_record = ConversationContextManager.create_job_record(
            content=jd,
            source="researched",
            title=role_title,
        )
        updated_job_history.append(job_record)

    return {
        "messages": [AIMessage(content=response_text)],
        "researched_jobs": researched_jobs,
        "job_descriptions_history": updated_job_history,
        "job_description": None
    }


def _perform_search_and_synthesis(job_title: str) -> Optional[str]:
    """Helper function to search and synthesize a job description with fallback."""
    print(f"--- Conducting research for: {job_title} ---")
    
    # Fallback job description template
    fallback_description = f"""
    **{job_title}**
    
    **Summary:**
    A {job_title} is responsible for contributing to the team's success through their technical expertise and collaborative approach.
    
    **Key Responsibilities:**
    • Develop and maintain high-quality solutions
    • Collaborate with cross-functional teams
    • Participate in planning and review processes
    • Continuously learn and adapt to new technologies
    
    **Required Skills:**
    • Strong technical foundation relevant to the role
    • Excellent communication and teamwork abilities
    • Problem-solving and analytical thinking
    • Adaptability and continuous learning mindset
    """
    
    # Check if search is configured
    searx_host = os.getenv("SEARX_HOST")
    if not searx_host:
        logger.warning("SEARX_HOST not configured, using fallback job description")
        return fallback_description
    
    search_queries = [
        f'responsibilities and skills for a "{job_title}" role',
        f'"day in the life of a {job_title}"',
        f'example job posting for "{job_title}"',
    ]

    try:
        wrapper = SearxSearchWrapper(searx_host=searx_host)
        all_results = []
        
        for query in search_queries:
            print(f"  - Querying: {query}")
            try:
                results = wrapper.results(query, num_results=3, categories=["general"])
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
        
        if not all_results:
            logger.warning("No search results found, using fallback")
            return fallback_description

        # Synthesize results
        synthesizer_llm = get_llm_with_fallback(model="llama-3.1-8b-instant", temperature=0.3)
        results_str = "\n".join(
            [
                f"Title: {res.get('title', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}"
                for res in all_results[:10]  # Limit results
            ]
        )
        
        synthesizer_prompt = f"""
        You are an expert research analyst. Synthesize the provided search results into a single, comprehensive job description for the role of '{job_title}'.
        Format it as a professional job description with sections for Summary, Key Responsibilities, and Required Skills.
        Base your synthesis SOLELY on the provided search results.
        Search results:
        ---
        {results_str[:3000]}  # Limit context
        ---
        """
        
        print("--- Synthesizing a unified job description... ---")
        result = call_llm_with_retry(synthesizer_llm, synthesizer_prompt)
        return result.content
        
    except Exception as e:
        logger.error(f"Failed to perform search and synthesis: {e}")
        return fallback_description



def search_node(state: GraphState):
    """
    A dedicated research node with context awareness.
    """
    print("\n--- Running Search Node (Context-Aware) ---")
    
    # Safely get state values
    try:
        user_request = state.get("original_user_request") or state.get("messages", [{}])[-1].content
        profile_data = state.get("profile_data", "")
        job_history = state.get("job_descriptions_history", [])
        current_context = state.get("current_job_context", [])
    except (IndexError, AttributeError) as e:
        logger.error(f"Failed to extract state values: {e}")
        return {
            "messages": [AIMessage(content="I need more context to help you. Could you please clarify what kind of job roles you're interested in?")],
            "plan": []
        }

    print("Search Node: Planner sent us here, so we search - no questions asked")

    try:
        intent_llm = get_llm_with_fallback(model="gemma2-9b-it")
        intent_prompt = f"""
        Analyze the user's request to determine their search-related intent.

        1. If the user asks for suggestions or what roles would suit them, the intent is 'suggest_roles'.
        2. If the user provides a specific job title to be analyzed or compared against, the intent is 'analyze_specific_role'.
        3. If the user has pasted what looks like a job description (with requirements, skills, experience sections), treat it as 'analyze_specific_role' with the title "Data Scientist" or extract the most relevant title from the description.

        Return ONLY a JSON object in this format:
        {{"intent": "suggest_roles" or "analyze_specific_role", "job_title": "title if analyze_specific_role else null"}}

        User Request: "{user_request[:1000]}"  # Limit request length
        """
        response = intent_llm.invoke(intent_prompt)
        import json
        
        # Extract content from response
        content = extract_llm_content(response)
        
        # Clean the JSON response
        content = clean_json_response(content)
        
        intent_dict = json.loads(content)
        intent_result = SearchIntent(**intent_dict)
        print(f"Search Intent: {intent_result.intent}")
    except Exception as e:
        logger.error(f"Failed to determine intent: {e}")
        # Default to role suggestions
        intent_result = SearchIntent(intent="suggest_roles", job_title=None)

    if intent_result.intent == "suggest_roles":
        print(
            "Search Node: Suggesting relevant job roles based on user's profile"
        )

        # Extract key info from profile for role suggestions
        profile_analyzer = get_llm_with_fallback(model="llama-3.3-70b-versatile", temperature=0.2)
        profile_analysis_prompt = f"""
        Analyze this LinkedIn profile to understand the person's background and experience level.
        
        Profile: {profile_data[:1500]}...
        
        Extract:
        1. Current role and experience level (0-2 years = Entry/Junior, 2-5 = Mid, 5+ = Senior)
        2. Core technical skills and technologies
        3. Domain expertise (e.g., AI/ML, web development, data science, etc.)
        4. Educational background
        5. Any specialized interests or focus areas
        
        Be specific and accurate based on the actual profile content.
        """

        profile_analysis = profile_analyzer.invoke(profile_analysis_prompt).content
        print(f"Search Node: Profile analysis completed")

        # Use the fallback function which now provides comprehensive role suggestions
        return _fallback_role_suggestions(profile_data, job_history)

    elif intent_result.intent == "analyze_specific_role":
        job_title = intent_result.job_title
        if not job_title:
            return {
                "messages": [
                    AIMessage(
                        content="I'm sorry, I couldn't identify a specific job title in your request. Could you please clarify?"
                    )
                ]
            }

        job_description = _perform_search_and_synthesis(job_title)
        if not job_description:
            return {
                "messages": [
                    AIMessage(
                        content=f"I'm sorry, I couldn't find enough information for '{job_title}'."
                    )
                ]
            }

        # Save this job to history as well
        job_record = ConversationContextManager.create_job_record(
            content=job_description, source="researched", title=job_title
        )
        updated_job_history = job_history.copy()
        updated_job_history.append(job_record)

        summary_message = AIMessage(
            content=f"I've researched the role of **{job_title}**. I will now proceed with the analysis."
        )
        return {
            "messages": [summary_message],
            "job_description": job_description,
            "job_descriptions_history": updated_job_history,
            "current_job_context": [job_record["id"]]
        }

    return {
        "messages": [
            AIMessage(
                content="I'm not sure how to handle that request. Could you rephrase?"
            )
        ]
    }



def job_fit_analyst_node(state: GraphState):
    """
    Compares the user's profile to a job description.
    """
    print("\n--- Running Job Fit Analyst ---")
    
    # Debug: Print state keys
    print(f"DEBUG: State keys available: {list(state.keys())}")
    
    # Robust error handling for missing job description
    job_description = state.get("job_description")
    profile_data = state.get("profile_data", "")
    
    print(f"DEBUG: job_description exists: {job_description is not None}")
    print(f"DEBUG: profile_data length: {len(profile_data) if profile_data else 0}")
    
    if not job_description:
        print("ERROR: No job description found in state")
        # Try to extract from the last message if it looks like a job description
        try:
            last_message = state.get("messages", [])[-1].content
            if len(last_message) > 200 and any(
                keyword in last_message.lower()
                for keyword in ["experience", "skills", "requirements"]
            ):
                print("Attempting to use last message as job description")
                job_description = last_message
            else:
                error_msg = (
                    "I need a job description to analyze your profile against. "
                    "Please either:\n"
                    "1. Paste a specific job description you'd like me to analyze\n"
                    "2. Ask me to research a specific role (e.g., 'Analyze my fit for a Senior Data Scientist role')\n"
                    "3. Ask for general role suggestions based on your profile"
                )
                return {"messages": [AIMessage(content=error_msg)], "plan": []}
        except (IndexError, AttributeError):
            return {
                "messages": [AIMessage(content="Please provide a job description for analysis.")],
                "plan": []
            }
    
    if not profile_data:
        return {
            "messages": [AIMessage(content="I don't have your profile data. Please start a new analysis session.")],
            "plan": []
        }

    try:
        # Truncate inputs if too long
        if len(profile_data) > MAX_PROFILE_LENGTH:
            profile_data = profile_data[:MAX_PROFILE_LENGTH]
        if len(job_description) > MAX_MESSAGE_LENGTH:
            job_description = job_description[:MAX_MESSAGE_LENGTH]
        
        # For Groq, we'll use JSON mode instead of structured output
        llm = get_llm_with_fallback(model="llama-3.3-70b-versatile")

        prompt = f"""
        Provide a detailed, actionable comparison between the provided LinkedIn profile and the target job description.
        Your analysis must be completely objective and based *only* on the text provided.

        IMPORTANT: Return ONLY the JSON object. Do not include any text before or after the JSON.
        Do not include markdown formatting. Do not include explanations.

        Return your response as valid JSON in the following format:
        {{
            "match_score": <0-100>,
            "score_reasoning": "<reasoning for the score>",
            "keyword_analysis": [
                {{"keyword": "<keyword>", "is_present": true/false, "reasoning": "<why present/absent>"}}
            ],
            "strengths": ["<strength 1>", "<strength 2>", ...],
            "gaps": ["<gap 1>", "<gap 2>", ...]
        }}

        **LinkedIn Profile:**
        ---
        {profile_data}
        ---
        **Target Job Description:**
        ---
        {job_description}
        ---
        """
        
        response = call_llm_with_retry(llm, prompt)
        
        # Parse JSON response
        import json
        try:
            # Extract content from response
            content = extract_llm_content(response)
            print(f"DEBUG: Response content type: {type(content)}")
            print(f"DEBUG: Response content (first 200 chars): {content[:200] if content else 'Empty'}")
            
            # Clean the JSON response
            content = clean_json_response(content)
            print(f"DEBUG: Cleaned JSON content (first 200 chars): {content[:200] if content else 'Empty'}")
            
            analysis_dict = json.loads(content)
            analysis = JobFitAnalysis(**analysis_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse structured response: {e}")
            logger.error(f"Raw content: {content if 'content' in locals() else 'No content extracted'}")
            raise
        
    except ValidationError as e:
        logger.error(f"Validation error in job fit analysis: {e}")
        # Fallback to unstructured response
        try:
            fallback_llm = get_llm_with_fallback(model="llama-3.1-8b-instant")
            fallback_prompt = f"""
            Compare this profile to the job description and provide:
            1. Overall match percentage (0-100)
            2. Key strengths (3-5 points)
            3. Gaps to address (3-5 points)
            
            Profile: {profile_data[:2000]}...
            Job Description: {job_description[:2000]}...
            """
            
            fallback_response = call_llm_with_retry(fallback_llm, fallback_prompt)
            return {"messages": [fallback_response], "plan": []}
            
        except Exception as e2:
            logger.error(f"Fallback analysis also failed: {e2}")
            return {
                "messages": [AIMessage(content="I'm having technical difficulties with the analysis. Please try again with a shorter job description.")],
                "plan": []
            }
    except Exception as e:
        logger.error(f"ERROR in job_fit_analyst: {e}")
        print(f"DEBUG: Exception in job_fit_analyst: {e}")
        print(f"DEBUG: Exception traceback: {traceback.format_exc()}")
        return {
            "messages": [
                AIMessage(
                    content="I encountered an error while analyzing the job fit. Please try rephrasing your request or ensure you've provided a valid job description."
                )
            ],
            "plan": []
        }

    markdown_output = f"# Job Fit Analysis\n\n"
    markdown_output += (
        f"## Overall Match Score: **{analysis.match_score}%**\n\n"
        f"*{analysis.score_reasoning}*\n\n"
    )
    markdown_output += "## Keyword Analysis\n\n"
    for keyword in analysis.keyword_analysis:
        status = "✅ Present" if keyword.is_present else "❌ Missing"
        markdown_output += f"• **{keyword.keyword}**: {status}\n"
    markdown_output += "\n## Your Strengths for this Role\n\n"
    for strength in analysis.strengths:
        markdown_output += f"• {strength}\n"
    markdown_output += "\n## Actionable Gaps & Suggestions\n\n"
    for gap in analysis.gaps:
        markdown_output += f"• {gap}\n"

    return {"messages": [AIMessage(content=markdown_output)]}



def content_enhancer_node(state: GraphState):
    """
    Rewrites a specific section of the user's profile.
    """
    print("\n--- Running Proactive Content Enhancer ---")
    
    # Safely get user request
    try:
        user_request = state.get("original_user_request") or state.get("messages", [])[-1].content
        profile_data = state.get("profile_data", "")
    except (IndexError, AttributeError) as e:
        logger.error(f"Failed to get user request: {e}")
        return {
            "messages": [AIMessage(content="Please specify which section of your profile you'd like me to enhance (headline, about, or experience).")],
            "plan": []
        }
    
    if not profile_data:
        return {
            "messages": [AIMessage(content="I don't have your profile data. Please start a new analysis session.")],
            "plan": []
        }
    
    llm = get_llm_with_fallback(model="llama-3.3-70b-versatile", temperature=0.7)

    sections_to_rewrite = []
    request_lower = user_request.lower()
    
    # Smart contextual detection using regex patterns
    import re
    
    # Pattern to detect section requests with context
    # Matches: "rewrite my about", "enhance the about section", "improve about section", etc.
    section_patterns = {
        "headline": [
            r'\b(?:rewrite|enhance|improve|update|modify|change)\s+(?:my\s+)?(?:the\s+)?headline',
            r'headline\s+(?:section|part)',
            r'\bmy\s+(?:professional\s+)?title\b'
        ],
        "about": [
            r'\b(?:rewrite|enhance|improve|update|modify|change)\s+(?:my\s+)?(?:the\s+)?about\s+(?:section)?',
            r'about\s+(?:section|part|me)',
            r'\b(?:my\s+)?(?:professional\s+)?(?:summary|overview)\b(?:\s+section)?'
        ],
        "experience": [
            r'\b(?:rewrite|enhance|improve|update|modify|change)\s+(?:my\s+)?(?:the\s+)?experience\s+(?:section)?',
            r'experience\s+(?:section|part)',
            r'\b(?:my\s+)?work\s+(?:history|experience)\b',
            r'\b(?:my\s+)?(?:job|employment)\s+(?:history|section)\b'
        ]
    }
    
    # Check each section's patterns
    for section, patterns in section_patterns.items():
        for pattern in patterns:
            if re.search(pattern, request_lower):
                sections_to_rewrite.append(section)
                break  # Avoid duplicate additions
    
    # If still no sections detected, look for section names immediately after action verbs
    if not sections_to_rewrite:
        # Look for patterns like "rewrite about" without "my" or "the"
        simple_patterns = {
            "headline": r'\b(?:rewrite|enhance|improve|update|modify|change|edit)\s+headline\b',
            "about": r'\b(?:rewrite|enhance|improve|update|modify|change|edit)\s+about\b',
            "experience": r'\b(?:rewrite|enhance|improve|update|modify|change|edit)\s+experience\b'
        }
        
        for section, pattern in simple_patterns.items():
            if re.search(pattern, request_lower):
                sections_to_rewrite.append(section)

    # Debug: Print what was detected
    print(f"Content Enhancer: User request: '{user_request[:100]}...'")
    print(f"Content Enhancer: Detected sections to rewrite: {sections_to_rewrite}")
    
    # If no specific section is mentioned, ask for clarification
    if not sections_to_rewrite:
        return {
            "messages": [AIMessage(content="I'd be happy to enhance your LinkedIn profile! Please specify which section(s) you'd like me to rewrite:\n\n• **Headline** - Your professional title and key expertise\n• **About** - Your professional summary and story\n• **Experience** - Your work history and achievements\n\nFor example: 'Please rewrite my about section' or 'Enhance my headline and experience'")],
            "plan": []
        }
    else:
        print(f"Identified sections for rewrite: {sections_to_rewrite}")

    # Truncate profile if too long
    if len(profile_data) > MAX_PROFILE_LENGTH:
        profile_data = profile_data[:MAX_PROFILE_LENGTH]
    
    prompt = f"""
    You are an expert LinkedIn copywriter and career strategist, speaking in the first person. Your tone is professional, confident, and results-oriented.

    I want you to enhance my LinkedIn profile based on my request. Here is your task:

    **IMPORTANT: Only rewrite the sections I specifically asked for: {', '.join(sections_to_rewrite)}**
    
    For each requested section, provide a rewritten, enhanced version based on the rules below. Present each rewritten section under a clear markdown heading (e.g., `## Enhanced About Section`).
        *   **For 'headline':** Create a concise, keyword-rich headline (under 220 characters) that includes my current role and key area of expertise (e.g., "Software Engineer at XYZ | Building with Generative AI & LLMs").
        *   **For 'about':** Write a compelling, first-person summary (under 150 words) that tells a story about my passion and impact.
        *   **For 'experience':** For my most recent job role, rewrite the description using 3-4 bullet points. Each bullet point **must be a single, flowing sentence that internally follows the STAR method (Situation, Task, Action, Result)** but **DO NOT** explicitly write the words "Situation:", "Task:", "Action:", or "Result:". For example, instead of "Situation: Faced with X... Task: Needed to Y... Action: I did Z... Result: Achieved A...", write "- Achieved A by implementing Z to solve the challenge of X, which was required for Y." Include quantifiable metrics.

    **My Full Profile:**
    ---
    {profile_data}
    ---
    
    Remember: ONLY provide rewrites for the sections I specifically asked for. Do not add any additional sections or proactive suggestions.

    **My Original Request:**
    ---
    "{user_request[:500]}"
    ---

    Provide only the rewritten sections as your response. Do not add any extra conversational text or commentary.
    """
    
    try:
        response = call_llm_with_retry(llm, prompt)
        return {"messages": [response], "plan": []}
    except Exception as e:
        logger.error(f"Failed to enhance content: {e}")
        # Provide a simpler fallback response
        fallback_msg = f"I'll help you enhance your {', '.join(sections_to_rewrite)} section(s).\n\n"
        fallback_msg += "Here are some general tips:\n"
        if "headline" in sections_to_rewrite:
            fallback_msg += "\n**Headline Tips:**\n- Include your current role and company\n- Add 2-3 key skills or specializations\n- Keep it under 220 characters\n"
        if "about" in sections_to_rewrite:
            fallback_msg += "\n**About Section Tips:**\n- Start with your passion or mission\n- Highlight your key achievements\n- End with what you're looking for\n"
        if "experience" in sections_to_rewrite:
            fallback_msg += "\n**Experience Tips:**\n- Use action verbs\n- Include quantifiable results\n- Focus on impact, not just duties\n"
        
        return {"messages": [AIMessage(content=fallback_msg)], "plan": []}



def career_counselor_node(state: GraphState):
    """
    Provides general career advice.
    """
    print("\n--- Running Career Counselor ---")
    
    # Safely get state values
    profile_data = state.get("profile_data", "")
    messages = state.get("messages", [])
    
    if not profile_data:
        return {
            "messages": [AIMessage(content="I need your profile information to provide personalized career advice. Please start a new analysis session.")],
            "plan": []
        }
    
    # Truncate inputs
    if len(profile_data) > MAX_PROFILE_LENGTH:
        profile_data = profile_data[:MAX_PROFILE_LENGTH]
    
    # Limit conversation history to recent messages
    recent_messages = messages[-5:] if len(messages) > 5 else messages
    messages_str = "\n".join([str(msg.content)[:500] if hasattr(msg, 'content') else str(msg)[:500] for msg in recent_messages])
    
    llm = get_llm_with_fallback(model="llama-3.3-70b-versatile", temperature=0.7)
    
    prompt = f"""
    You are a pragmatic, senior-level career coach. Your advice must be realistic and based on tangible steps.
    Use the user's profile and conversation history to inform your response, but **do not mention that you are reviewing them.** Respond directly to the user's query.

    **Your Task:**
    1.  First, determine if the user's stated or implied career goal is a significant leap from their current role (e.g., a junior developer asking to become a CTO).
    2.  **If it is a leap:** Acknowledge their ambition positively, but then ground the conversation in reality. Outline a realistic, step-by-step career ladder they would need to climb. Focus ALL of your advice on achieving the **very next step** on that ladder.
    3.  **If the goal is a reasonable next step:** Provide a targeted skill gap analysis, suggest 2-3 specific, high-impact learning resources, and offer strategic advice on networking and project selection for that role.
    
    Here is the necessary context:
    **User Profile:** --- {profile_data} ---
    **Recent Conversation:** --- {messages_str} ---
    """
    
    try:
        response = call_llm_with_retry(llm, prompt)
        return {"messages": [response], "plan": []}
    except Exception as e:
        logger.error(f"Failed to provide career advice: {e}")
        # Provide generic but helpful advice
        generic_advice = """
Based on our conversation, here are some general career development tips:

1. **Continuous Learning**: Stay updated with industry trends and technologies
2. **Networking**: Build connections in your field through LinkedIn and professional events
3. **Portfolio Building**: Work on projects that demonstrate your skills
4. **Mentorship**: Seek guidance from professionals in your target role
5. **Skill Assessment**: Identify gaps between your current skills and your target role

Would you like me to elaborate on any of these areas?
        """
        return {"messages": [AIMessage(content=generic_advice)], "plan": []}



def planner_node(state: GraphState):
    """
    The master planner with production-grade context awareness.
    Handles multiple JDs, comparisons, references, and complex user flows.
    """
    print("\n--- Running Master Planner (Production Grade) ---")

    try:
        # Safely extract state values
        messages = state.get("messages", [])
        if not messages:
            return {
                "plan": ["career_counselor"],
                "messages": [AIMessage(content="How can I help you with your LinkedIn profile and career development?")]
            }
        
        user_request = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        profile_data = state.get("profile_data", "")
        job_history = state.get("job_descriptions_history", [])
        recent_messages = [
            msg.content for msg in messages[-5:] if hasattr(msg, "content")
        ]

        # Validate inputs
        if len(user_request) > MAX_MESSAGE_LENGTH:
            user_request = user_request[:MAX_MESSAGE_LENGTH]

        # Use the context manager to understand intent
        try:
            intent = ConversationContextManager.understand_user_intent(
                user_request, job_history, recent_messages
            )
        except Exception as e:
            logger.error(f"Failed to understand intent: {e}")
            # Default intent
            intent = {"type": "general", "details": {}}

        print(f"Planner: Detected intent type: {intent['type']}")
        print(f"Planner: Intent details: {intent.get('details', {})}")

        # Initialize return state updates
        state_updates = {
            "plan": [],
            "original_user_request": user_request,
            "conversation_context": state.get("conversation_context", {}),
            "planner_decision": intent,  # Pass the planner's decision to other nodes
        }
        state_updates["conversation_context"]["last_intent"] = intent

        # Handle different intent types
        if intent["type"] == "pasted_jd":
            # User pasted a job description
            job_record = ConversationContextManager.create_job_record(
                content=user_request, source="pasted"
            )

            # Update job history
            updated_history = state.get("job_descriptions_history", []).copy()
            updated_history.append(job_record)
            state_updates["job_descriptions_history"] = updated_history

            # Set as current job
            state_updates["job_description"] = user_request
            state_updates["current_job_context"] = [job_record["id"]]

            # Determine tasks
            tasks = ["job_fit_analyst"]
            if any(
                kw in user_request.lower()
                for kw in ["enhance", "rewrite", "improve", "about"]
            ):
                tasks.append("content_enhancer")

            state_updates["plan"] = tasks
            print(
                f"Planner: Handling pasted JD '{job_record['title']}' with tasks: {tasks}"
            )

        elif intent["type"] == "comparison":
            # User wants to compare multiple jobs
            job_ids = intent.get("job_ids", [])
            if len(job_ids) < 2:
                # Not enough jobs to compare
                state_updates["plan"] = ["career_counselor"]
                state_updates["conversation_context"][
                    "error"
                ] = "Need at least 2 jobs to compare"
            else:
                # For now, use career counselor for comparisons
                # Later we can add a dedicated comparison node
                state_updates["plan"] = ["career_counselor"]
                state_updates["current_job_context"] = job_ids

        elif intent["type"] == "reference":
            # User referring to a previous job
            job_id = intent.get("job_id")
            if job_id:
                # Find the referenced job
                for job in job_history:
                    if job["id"] == job_id:
                        state_updates["job_description"] = job["content"]
                        state_updates["current_job_context"] = [job_id]
                        state_updates["plan"] = ["job_fit_analyst"]
                        print(f"Planner: Using referenced job '{job['title']}'")
                        break
            else:
                state_updates["plan"] = ["career_counselor"]

        elif intent["type"] == "follow_up":
            # Follow-up question about analyzed jobs
            job_ids = intent.get("job_ids", [])
            if job_ids:
                state_updates["current_job_context"] = job_ids
            state_updates["plan"] = ["career_counselor"]

        elif intent["type"] == "new_analysis":
            # New job analysis request
            state_updates["plan"] = ["search", "job_fit_analyst"]

            # Check if enhancement is also requested
            if any(
                kw in user_request.lower() for kw in ["enhance", "rewrite", "improve"]
            ):
                state_updates["plan"].append("content_enhancer")

        else:  # general request - use smart LLM planning
            planner_llm = get_llm_with_fallback(model="llama-3.3-70b-versatile")

            context_summary = ConversationContextManager.get_context_summary(
                job_history
            )
            recent_messages = [
                msg.content for msg in state["messages"][-3:] if hasattr(msg, "content")
            ]

            prompt = f"""
            You are an expert AI task planner. Analyze the user's request and create an intelligent plan.
            
            **Conversation History:**
            {context_summary}
            
            **Recent Messages:**
            {recent_messages}
            
            **Current User Request:**
            "{user_request}"
            
            **Available Tools:**
            - `search`: ALWAYS use for new job roles, role suggestions, or any job-related research
            - `job_fit_analyst`: Use ONLY after search when user wants analysis/comparison  
            - `content_enhancer`: Use for profile rewriting/enhancement requests
            - `career_counselor`: Use ONLY for follow-up questions, advice, certifications, skills
            
            **Key Rules:**
            1. If user asks "what roles suit me" or similar → [`search`] ONLY (no job_fit_analyst)
            2. If user provides new JD or job role → [`search`, `job_fit_analyst`]  
            3. If user asks follow-up questions about previous analysis → [`career_counselor`] ONLY
            4. If user wants both analysis + profile rewrite → [`search`, `job_fit_analyst`, `content_enhancer`]
            5. If user asks about skills/certifications for a role we discussed → [`career_counselor`] ONLY
            
            BE SMART: Don't use job_fit_analyst unless user specifically wants job analysis/comparison.
            Role suggestions = search only!
            """

            # Add JSON format to prompt
            prompt += """
            
            Return ONLY a JSON object in this format:
            {"tasks": ["search", "job_fit_analyst", "content_enhancer", "career_counselor"]}
            Include only the tasks needed.
            """
            
            response = planner_llm.invoke(prompt)
            import json
            
            # Extract content from response
            content = extract_llm_content(response)
            
            # Clean the JSON response
            content = clean_json_response(content)
            
            plan_dict = json.loads(content)
            state_updates["plan"] = plan_dict["tasks"]
            print(f"Planner: Smart LLM generated plan: {plan_dict['tasks']}")

        # Safety validations
        if "job_fit_analyst" in state_updates["plan"] and not state_updates.get(
            "job_description"
        ):
            # Need a job description for analysis
            if not state.get("current_job_context") and not state.get(
                "job_descriptions_history"
            ):
                # No context available, add search
                if "search" not in state_updates["plan"]:
                    idx = state_updates["plan"].index("job_fit_analyst")
                    state_updates["plan"].insert(idx, "search")
                    print("Planner: Added 'search' before 'job_fit_analyst' for safety")

        # Legacy compatibility: if old code expects job_description, ensure it's set
        if not state_updates.get("job_description") and state.get("job_description"):
            state_updates["job_description"] = state["job_description"]

        return state_updates

    except Exception as e:
        logger.error(f"ERROR in planner_node: {e}\n{traceback.format_exc()}")
        # Graceful fallback
        return {
            "plan": ["career_counselor"],
            "original_user_request": state.get("messages", [{}])[-1].content if state.get("messages") else "",
            "conversation_context": {"error": str(e)}
        }



def plan_executor_node(state: GraphState):
    """
    The central control unit. Updates the plan and routes to the next task.
    Includes fallback response mechanism to prevent silent failures.
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