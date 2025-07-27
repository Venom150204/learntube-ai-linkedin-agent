from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
import sqlite3

from src.utils import scrape_linkedin_profile
from src.graph import create_workflow
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# Load environment variables
load_dotenv()

# --- Pydantic Models for API validation ---
class StartRequest(BaseModel):
    linkedin_url: str

class ChatRequest(BaseModel):
    message: str

# --- FastAPI App Setup ---
app = FastAPI()

# Add CORS middleware for Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Streamlit Cloud URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health Check ---
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "LinkedIn Profile Optimizer API"}

# --- Memory/Checkpointer Setup ---
# This is a robust way to handle DB connections in a web server
conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
memory = SqliteSaver(conn=conn)
workflow = create_workflow(checkpointer=memory)

# --- API Endpoints ---
@app.post("/start_analysis")
async def start_analysis(request: StartRequest):
    """
    Starts a new analysis and chat session.
    Scrapes the profile, initializes the graph, and returns a new thread_id.
    """
    try:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"[{thread_id}] Scraping profile: {request.linkedin_url}")
        profile_data = scrape_linkedin_profile(request.linkedin_url)
        print(f"[{thread_id}] Scraping complete.")

        initial_state = {
            "messages": [HumanMessage(content=f"Analyzing profile: {request.linkedin_url}")],
            "profile_data": profile_data
        }
        
        print(f"[{thread_id}] Invoking workflow for initial analysis...")
        workflow.invoke(initial_state, config)
        print(f"[{thread_id}] Initial analysis complete.")
        
        return {"thread_id": thread_id}
    except Exception as e:
        print(f"Error in /start_analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{thread_id}")
async def chat(thread_id: str, request: ChatRequest):
    """
    Handles a message within an existing chat session.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"[{thread_id}] Received message: {request.message}")
        
        # Invoke the workflow with the new message
        input_message = {"messages": [HumanMessage(content=request.message)]}
        workflow.invoke(input_message, config)
        
        print(f"[{thread_id}] Workflow invocation complete.")
        
        # Get the latest state to return the new messages
        latest_state = workflow.get_state(config)
        return {"messages": latest_state.values.get("messages", [])}
        
    except Exception as e:
        print(f"Error in /chat/{thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """
    Retrieves the full chat history for a session.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = workflow.get_state(config)
        return {"messages": state.values.get("messages", [])}
    except Exception as e:
        print(f"Error in /history/{thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

