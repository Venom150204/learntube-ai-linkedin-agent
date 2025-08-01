import streamlit as st
import requests
import time

# --- Configuration ---
import os
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- Helper Functions to interact with the API ---

def start_analysis(linkedin_url: str):
    """Calls the /start_analysis endpoint and returns the new thread_id."""
    response = requests.post(f"{API_BASE_URL}/start_analysis", json={"linkedin_url": linkedin_url})
    
    # Handle validation errors specifically
    if response.status_code == 422:
        error_detail = response.json().get("detail", [])
        if isinstance(error_detail, list) and error_detail:
            # Pydantic validation error format
            error_msg = error_detail[0].get("msg", "Invalid input")
        else:
            # Direct error message
            error_msg = error_detail
        raise ValueError(error_msg)
    
    response.raise_for_status() # Will raise an exception for other 4XX/5XX errors
    return response.json()["thread_id"]

def post_message(thread_id: str, message: str):
    """Calls the /chat/{thread_id} endpoint."""
    response = requests.post(f"{API_BASE_URL}/chat/{thread_id}", json={"message": message})
    response.raise_for_status()
    return response.json()

def get_history(thread_id: str):
    """Calls the /history/{thread_id} endpoint."""
    response = requests.get(f"{API_BASE_URL}/history/{thread_id}")
    response.raise_for_status()
    return response.json().get("messages", [])

# --- Streamlit UI ---

def main():
    st.title("LinkedIn Profile Optimizer")

    # Manage session state
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    
    # 1. Initial Screen: Enter LinkedIn URL
    if not st.session_state.thread_id:
        linkedin_url = st.text_input(
            "Enter your LinkedIn Profile URL or username", 
            placeholder="e.g., dhruv-patel7/ or https://linkedin.com/in/dhruv-patel7/"
        )
        if st.button("Start Analysis"):
            if linkedin_url:
                with st.spinner("Starting analysis... This may take a moment."):
                    try:
                        thread_id = start_analysis(linkedin_url)
                        st.session_state.thread_id = thread_id
                        # Wait a moment for the initial analysis to be available
                        time.sleep(5) 
                        st.rerun()
                    except ValueError as e:
                        # Handle validation errors - just show the error once
                        st.error(str(e))
                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to connect to the backend: {e}")
            else:
                st.warning("Please provide a LinkedIn URL.")
        return # Stop further execution until we have a thread_id

    # 2. Main Chat Interface
    # Display the chat history
    try:
        messages = get_history(st.session_state.thread_id)
        for msg in messages:
            # The API returns dicts, not objects, so we access keys directly
            with st.chat_message(msg.get("type", "human")):
                st.markdown(msg.get("content", ""))
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch chat history: {e}")
        return

    # Handle new user input
    if prompt := st.chat_input("Ask a follow-up question..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            try:
                post_message(st.session_state.thread_id, prompt)
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to send message: {e}")

if __name__ == "__main__":
    main() 