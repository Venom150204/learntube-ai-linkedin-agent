# LearnTube.ai - AI-Powered LinkedIn Profile Optimizer

This project is an AI-powered chat system that helps users optimize their LinkedIn profiles, analyze job fit, and provide career guidance.

## Features

- Interactive chat interface for profile optimization.
- LinkedIn profile analysis to identify areas for improvement.
- Job fit analysis with a match score and suggestions.
- AI-powered content enhancement for profile sections.
- Career counseling and skill gap analysis.

## Tech Stack

- **Framework:** Streamlit
- **AI Orchestration:** LangGraph
- **LinkedIn Scraping:** Apify
- **LLM:** OpenAI GPT-4o
- **Memory:** LangGraph Checkpointers with SQLite

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/learntube-ai-profiler.git
    cd learntube-ai-profiler
    ```

2.  Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  Create a `.env` file and add your API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    APIFY_API_KEY="your_apify_api_key_here"
    LANGCHAIN_TRACING_V2="true"
    LANGCHAIN_API_KEY="your_langchain_api_key_here"
    ```

4.  Run the application:

    This is now a two-part application. You need to run the backend API server first in one terminal, and then run the Streamlit UI in a second terminal.

    **Terminal 1: Run the FastAPI Backend**
    ```bash
    python api.py
    ```
    You will see logs in this terminal.

    **Terminal 2: Run the Streamlit Frontend**
    ```bash
    streamlit run app.py
    ```
    This will open the chat interface in your browser.

## Architecture

This application is built using a client-server architecture.

### Core Components:

1.  **FastAPI Backend (`api.py`)**: This is the core of the application, containing all the business logic. It exposes a REST API for the frontend to interact with.
    *   **LangGraph Workflow**: The multi-agent system, router, and all AI logic reside here.
    *   **Checkpointer (Memory)**: The backend manages the `SqliteSaver` checkpointer for persistent, session-based memory.
    *   **Endpoints**: It provides endpoints to start a new chat, send a message, and get the history.

2.  **Streamlit Frontend (`app.py`)**: This provides the interactive chat interface for the user. It is a "dumb" client that simply makes HTTP requests to the FastAPI backend. It manages a `thread_id` to maintain the session with the backend.

### How it Works:

1.  The user starts the FastAPI server.
2.  The user starts the Streamlit app, which opens in the browser.
3.  When the user enters a LinkedIn URL in the UI, the Streamlit app sends a `POST` request to the `/start_analysis` endpoint on the backend.
4.  The backend scrapes the profile, runs the initial analysis via LangGraph, and returns a unique `thread_id` to the frontend.
5.  The Streamlit app saves this `thread_id` and uses it for all future requests.
6.  When the user sends a message, the UI makes a `POST` request to the `/chat/{thread_id}` endpoint. The backend processes the message through the LangGraph workflow and saves the state.
7.  The UI periodically makes `GET` requests to `/history/{thread_id}` to refresh the chat display.

## Challenges and Solutions

*   **Challenge**: Ensuring the conversation is natural and continuous, rather than a series of one-off interactions.
    *   **Solution**: The graph is designed as a loop. After each agent completes its task, the flow is directed back to the central router, which then waits for the next user input. This creates a seamless, back-and-forth conversational experience.

*   **Challenge**: Avoiding unnecessary LLM calls and token wastage by sending irrelevant context to agents.
    *   **Solution**: We implemented two key strategies:
        1.  **Conditional Routing**: Only the agent best suited for the user's request is ever called.
        2.  **Contextual Prompt Engineering**: Each agent's prompt is carefully crafted to only include the data it needs from the `GraphState`. For example, the `job_fit_analyst` only receives the profile data and the job description, not the entire chat history.

*   **Challenge**: Maintaining context across multiple user sessions and interactions.
    *   **Solution**: We leveraged LangGraph's built-in `SqliteSaver` checkpointer. It automatically saves the entire graph state after each step, tied to a unique session ID. This provides robust, persistent memory with minimal manual setup. 