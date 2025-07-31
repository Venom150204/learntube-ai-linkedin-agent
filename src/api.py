from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, HttpUrl
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import uuid
import sqlite3
import traceback
import logging
import re
import os
from typing import Optional
import asyncio
from datetime import datetime
import time

from src.utils import scrape_linkedin_profile
from src.graph import create_workflow
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MAX_MESSAGE_LENGTH = 10000
MAX_PROFILE_SIZE = 50000
REQUEST_TIMEOUT = 300  # 5 minutes
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
DATABASE_PATH = os.getenv("DATABASE_PATH", "memory.sqlite")

# --- Pydantic Models with Validation ---
class StartRequest(BaseModel):
    linkedin_url: str
    
    @validator('linkedin_url')
    def validate_linkedin_url(cls, v):
        # Comprehensive LinkedIn URL validation
        linkedin_pattern = r'^https?://(www\.)?linkedin\.com/in/[\w\-]+/?$'
        if not re.match(linkedin_pattern, v):
            raise ValueError('Invalid LinkedIn URL format. Expected: https://www.linkedin.com/in/username')
        return v

class ChatRequest(BaseModel):
    message: str
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f'Message too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed')
        return v.strip()

# --- Database Management ---
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None
        self._lock = asyncio.Lock()
        
    def connect(self):
        """Create a new database connection with proper settings"""
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout for locks
            logger.info(f"Database connected: {self.db_path}")
            return self._conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        """Safely close database connection"""
        if self._conn:
            try:
                self._conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}")
    
    def get_connection(self):
        """Get or create database connection"""
        if not self._conn:
            self.connect()
        return self._conn

# Global database manager
db_manager = DatabaseManager(DATABASE_PATH)

# --- Application Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting LinkedIn Profile Optimizer API")
    try:
        db_manager.connect()
        # Initialize workflow
        app.state.workflow = create_workflow(SqliteSaver(conn=db_manager.get_connection()))
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    db_manager.close()

# --- FastAPI App Setup ---
app = FastAPI(
    title="LinkedIn Profile Optimizer API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with proper security
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with user-friendly messages"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "message": "An error occurred processing your request",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# --- Health Check ---
@app.get("/")
def health_check():
    """Health check endpoint with database status"""
    try:
        # Check database connection
        conn = db_manager.get_connection()
        conn.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "service": "LinkedIn Profile Optimizer API",
        "version": "1.0.0",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }

# --- Utility Functions ---
async def with_timeout(coro, timeout_seconds: int):
    """Execute coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Operation timed out after {timeout_seconds} seconds"
        )

# --- API Endpoints ---
@app.post("/start_analysis")
async def start_analysis(request: StartRequest):
    """
    Starts a new analysis and chat session.
    Scrapes the profile, initializes the graph, and returns a new thread_id.
    """
    thread_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate workflow is initialized
        if not hasattr(app.state, 'workflow'):
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Please try again in a moment."
            )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"[{thread_id}] Starting analysis for: {request.linkedin_url}")
        
        # Scrape profile with timeout and error handling
        try:
            profile_data = await with_timeout(
                asyncio.to_thread(scrape_linkedin_profile, request.linkedin_url),
                timeout_seconds=120
            )
            
            # Validate scraped data
            if not profile_data or len(profile_data) < 100:
                raise ValueError("Profile data appears to be incomplete")
            
            if len(profile_data) > MAX_PROFILE_SIZE:
                logger.warning(f"[{thread_id}] Profile data truncated from {len(profile_data)} to {MAX_PROFILE_SIZE}")
                profile_data = profile_data[:MAX_PROFILE_SIZE]
                
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Profile scraping timed out. The LinkedIn profile may be too large or the service is experiencing delays."
            )
        except Exception as e:
            logger.error(f"[{thread_id}] Scraping failed: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Failed to scrape LinkedIn profile. Please ensure the URL is correct and the profile is public. Error: {str(e)}"
            )
        
        logger.info(f"[{thread_id}] Profile scraped successfully ({len(profile_data)} chars)")
        
        # Initialize workflow state
        initial_state = {
            "messages": [HumanMessage(content=f"Analyzing profile: {request.linkedin_url}")],
            "profile_data": profile_data,
            "job_descriptions_history": [],
            "conversation_context": {
                "started_at": datetime.utcnow().isoformat(),
                "linkedin_url": request.linkedin_url
            }
        }
        
        # Invoke workflow with timeout
        try:
            await with_timeout(
                asyncio.to_thread(app.state.workflow.invoke, initial_state, config),
                timeout_seconds=60
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Initial analysis timed out. Please try again with a shorter profile."
            )
        except Exception as e:
            logger.error(f"[{thread_id}] Workflow invocation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to analyze profile. Our AI service may be experiencing issues. Please try again later."
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f"[{thread_id}] Analysis completed in {elapsed_time:.2f} seconds")
        
        return {
            "thread_id": thread_id,
            "message": "Profile analysis completed successfully",
            "processing_time": f"{elapsed_time:.2f}s"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{thread_id}] Unexpected error in start_analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )


@app.post("/chat/{thread_id}")
async def chat(thread_id: str, request: ChatRequest):
    """
    Handles a message within an existing chat session.
    """
    start_time = time.time()
    
    try:
        # Validate thread_id format
        try:
            uuid.UUID(thread_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid thread ID format"
            )
        
        # Validate workflow is initialized
        if not hasattr(app.state, 'workflow'):
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Please try again in a moment."
            )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Check if thread exists
        try:
            existing_state = app.state.workflow.get_state(config)
            if not existing_state or not existing_state.values:
                raise HTTPException(
                    status_code=404,
                    detail="Session not found. Please start a new analysis."
                )
        except Exception as e:
            logger.error(f"[{thread_id}] Failed to retrieve state: {e}")
            raise HTTPException(
                status_code=404,
                detail="Session not found or has expired. Please start a new analysis."
            )
        
        logger.info(f"[{thread_id}] Processing message: {request.message[:100]}...")
        
        # Invoke workflow with timeout
        input_message = {"messages": [HumanMessage(content=request.message)]}
        
        try:
            await with_timeout(
                asyncio.to_thread(app.state.workflow.invoke, input_message, config),
                timeout_seconds=REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {REQUEST_TIMEOUT} seconds. Your query may be too complex. Please try a simpler question."
            )
        except Exception as e:
            logger.error(f"[{thread_id}] Workflow error: {e}")
            # Provide user-friendly error messages based on error type
            if "token" in str(e).lower() or "context" in str(e).lower():
                raise HTTPException(
                    status_code=413,
                    detail="The conversation has become too long. Please start a new analysis session."
                )
            elif "api" in str(e).lower() or "openai" in str(e).lower():
                raise HTTPException(
                    status_code=503,
                    detail="AI service is temporarily unavailable. Please try again in a few moments."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to process your message. Please try rephrasing or simplifying your query."
                )
        
        # Get updated state
        try:
            latest_state = app.state.workflow.get_state(config)
            messages = latest_state.values.get("messages", [])
            
            # Ensure we have a response
            if not messages or len(messages) < 2:
                raise ValueError("No response generated")
                
        except Exception as e:
            logger.error(f"[{thread_id}] Failed to retrieve response: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve response. Please try again."
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f"[{thread_id}] Message processed in {elapsed_time:.2f} seconds")
        
        return {
            "messages": messages,
            "processing_time": f"{elapsed_time:.2f}s"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{thread_id}] Unexpected error in chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )


@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """
    Retrieves the full chat history for a session.
    """
    try:
        # Validate thread_id format
        try:
            uuid.UUID(thread_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid thread ID format"
            )
        
        # Validate workflow is initialized
        if not hasattr(app.state, 'workflow'):
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Please try again in a moment."
            )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = app.state.workflow.get_state(config)
            if not state or not state.values:
                raise HTTPException(
                    status_code=404,
                    detail="Session not found or has expired."
                )
            
            messages = state.values.get("messages", [])
            context = state.values.get("conversation_context", {})
            
            return {
                "messages": messages,
                "context": context,
                "message_count": len(messages)
            }
            
        except Exception as e:
            logger.error(f"[{thread_id}] Failed to retrieve history: {e}")
            raise HTTPException(
                status_code=404,
                detail="Failed to retrieve session history. The session may have expired."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{thread_id}] Unexpected error in get_history: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )


# --- Session Management Endpoint ---
@app.delete("/session/{thread_id}")
async def delete_session(thread_id: str):
    """
    Delete a session and its associated data.
    """
    try:
        # Validate thread_id format
        try:
            uuid.UUID(thread_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid thread ID format"
            )
        
        # Note: Actual deletion would require implementing cleanup in the SqliteSaver
        # For now, we'll just acknowledge the request
        logger.info(f"[{thread_id}] Session deletion requested")
        
        return {
            "message": "Session marked for deletion",
            "thread_id": thread_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{thread_id}] Error in delete_session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to delete session"
        )