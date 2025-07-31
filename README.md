# LearnTube.ai - AI-Powered LinkedIn Profile Optimizer

This project is an AI-powered chat system that helps users optimize their LinkedIn profiles, analyze job fit, and provide career guidance.

## Features

- Interactive chat interface for profile optimization.
- LinkedIn profile analysis to identify areas for improvement.
- Job fit analysis with a match score and suggestions.
- AI-powered content enhancement for profile sections.
- Career counseling and skill gap analysis.
- **Multi-Job Context Awareness**: Analyze and compare multiple job opportunities in a single conversation.
- **Smart Reference Resolution**: Reference previously discussed jobs naturally (e.g., "compare the first job with this one").
- **Enhanced Job Description Detection**: Automatically recognizes when you paste a job description.
- **Improved Multi-Task Planning**: Handle complex requests that require multiple analysis steps.
- **Production-Grade Error Handling**: Robust error recovery and informative error messages.
- **Sequential Task Execution**: Fixed workflow to properly execute multiple tasks in sequence (e.g., job analysis followed by content enhancement).

## Tech Stack

- **Framework:** Streamlit
- **AI Orchestration:** LangGraph
- **LinkedIn Scraping:** Apify
- **LLM:** Groq (using Llama 3.3 70B for main tasks, Llama 3.1 8B for fast operations)
- **Memory:** LangGraph Checkpointers with SQLite
- **Backend:** FastAPI
- **State Management:** Enhanced GraphState with job history tracking
- **Project Structure:** Modularized architecture with separate `src/` directory

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
    # Groq Configuration (Primary LLM)
    GROQ_API_KEY="your_groq_api_key_here"
    
    # LinkedIn Scraping
    APIFY_API_KEY="your_apify_api_key_here"
    
    # Search functionality (optional)
    SEARX_HOST='your_searx_instance_url'
    
    # LangChain Tracing (optional)
    LANGCHAIN_TRACING_V2="true"
    LANGCHAIN_API_KEY="your_langchain_api_key_here"
    ```

4.  Run the application:

    This is now a two-part application. You need to run the backend API server first in one terminal, and then run the Streamlit UI in a second terminal.

    **Terminal 1: Run the FastAPI Backend**
    ```bash
    python start.py
    ```
    You will see logs in this terminal. The backend will run on port 8000 by default.

    **Terminal 2: Run the Streamlit Frontend**
    ```bash
    streamlit run app.py
    ```
    This will open the chat interface in your browser.

## Architecture

This application is built using a client-server architecture with an advanced context-aware AI system and modularized codebase.

### Project Structure:
```
LearnTube.ai/
├── src/
│   ├── api.py              # FastAPI backend server
│   ├── graph.py            # LangGraph workflow definition
│   ├── utils.py            # Utility functions (scraping, etc.)
│   ├── context_manager.py  # Conversation context management
│   ├── core/
│   │   ├── error_handling.py  # Error handling decorators
│   │   └── llm_utils.py       # LLM configuration and utilities
│   ├── models/
│   │   └── schemas.py      # Pydantic models and GraphState
│   └── nodes/
│       └── nodes.py        # All agent nodes in one file
├── app.py                  # Streamlit frontend
├── start.py               # Backend startup script
└── requirements.txt       # Project dependencies
```

### Core Components:

1.  **FastAPI Backend (`src/api.py`)**: This is the core of the application, containing all the business logic. It exposes a REST API for the frontend to interact with.
    *   **LangGraph Workflow**: The multi-agent system, router, and all AI logic reside here.
    *   **Checkpointer (Memory)**: The backend manages the `SqliteSaver` checkpointer for persistent, session-based memory.
    *   **Endpoints**: It provides endpoints to start a new chat, send a message, and get the history.
    *   **Enhanced Error Handling**: Comprehensive error handling with proper logging.

2.  **Streamlit Frontend (`app.py`)**: This provides the interactive chat interface for the user. It is a "dumb" client that simply makes HTTP requests to the FastAPI backend. It manages a `thread_id` to maintain the session with the backend.

3.  **Context Manager (`src/context_manager.py`)**: A sophisticated conversation context management system that:
    *   **Intent Recognition**: Intelligently understands user intent across multiple conversation turns
    *   **Job History Tracking**: Maintains a history of all analyzed job descriptions with unique IDs
    *   **Reference Resolution**: Handles references to previously discussed jobs (e.g., "the previous role", "that data scientist position")
    *   **Comparison Support**: Detects when users want to compare multiple job opportunities
    *   **Smart JD Detection**: Automatically identifies when users paste job descriptions vs. asking questions

### How it Works:

1.  The user starts the FastAPI server.
2.  The user starts the Streamlit app, which opens in the browser.
3.  When the user enters a LinkedIn URL in the UI, the Streamlit app sends a `POST` request to the `/start_analysis` endpoint on the backend.
4.  The backend scrapes the profile, runs the initial analysis via LangGraph, and returns a unique `thread_id` to the frontend.
5.  The Streamlit app saves this `thread_id` and uses it for all future requests.
6.  When the user sends a message, the UI makes a `POST` request to the `/chat/{thread_id}` endpoint. The backend processes the message through the LangGraph workflow and saves the state.
7.  The UI periodically makes `GET` requests to `/history/{thread_id}` to refresh the chat display.

### Enhanced Workflow (Graph System):

The application now uses an advanced graph-based workflow system (`src/graph.py`) with the following improvements:

1. **Master Planner Node**: 
   - Production-grade intent understanding using the Context Manager
   - Creates multi-step execution plans based on user requests
   - Handles complex scenarios like job comparisons and references

2. **Plan Executor Node**: 
   - Executes tasks sequentially based on the planner's output
   - Manages state transitions between different agents
   - Ensures robust error handling at each step

3. **Enhanced Agent Nodes**:
   - **Search Node**: Context-aware job searching with caching
   - **Job Fit Analyst**: Improved error handling and fallback mechanisms
   - **Content Enhancer**: Profile section rewriting capabilities
   - **Career Counselor**: General career advice with context awareness

4. **State Management**:
   - Tracks job description history with unique IDs
   - Maintains current job context for follow-up questions
   - Stores conversation context for intelligent responses

## Challenges and Solutions

*   **Challenge**: Ensuring the conversation is natural and continuous, rather than a series of one-off interactions.
    *   **Solution**: The graph is designed as a loop. After each agent completes its task, the flow is directed back to the central router, which then waits for the next user input. This creates a seamless, back-and-forth conversational experience.

*   **Challenge**: Avoiding unnecessary LLM calls and token wastage by sending irrelevant context to agents.
    *   **Solution**: We implemented two key strategies:
        1.  **Conditional Routing**: Only the agent best suited for the user's request is ever called.
        2.  **Contextual Prompt Engineering**: Each agent's prompt is carefully crafted to only include the data it needs from the `GraphState`. For example, the `job_fit_analyst` only receives the profile data and the job description, not the entire chat history.

*   **Challenge**: Maintaining context across multiple user sessions and interactions.
    *   **Solution**: We leveraged LangGraph's built-in `SqliteSaver` checkpointer. It automatically saves the entire graph state after each step, tied to a unique session ID. This provides robust, persistent memory with minimal manual setup.

*   **Challenge**: Handling complex multi-task requests in a single query.
    *   **Solution**: Implemented a Master Planner and Plan Executor architecture that:
        - Breaks down complex requests into sequential tasks
        - Maintains execution state between tasks
        - Provides clear error handling and recovery

*   **Challenge**: Managing multiple job descriptions and references in conversations.
    *   **Solution**: Created a sophisticated Context Manager that:
        - Assigns unique IDs to each job description
        - Tracks job history throughout the conversation
        - Intelligently resolves references like "the previous job" or "that data scientist role"
        - Supports comparisons between multiple jobs

## Recent Updates and Improvements

### LLM Migration to Groq (2025-07-31)
- **Replaced OpenAI with Groq**: Migrated from OpenAI GPT models to Groq for better performance and cost efficiency
- **Models Used**:
  - `llama-3.3-70b-versatile`: Primary model for complex tasks (profile analysis, job fit, content enhancement)
  - `llama-3.1-8b-instant`: Fast model for simple operations and fallback scenarios
  - `gemma2-9b-it`: Used for structured output tasks
- **JSON Mode**: Implemented JSON response parsing instead of function calling for better Groq compatibility

### Critical Bug Fixes
1. **Sequential Task Execution**: Fixed workflow bug where content_enhancer wasn't being called after job_fit_analyst
   - Root cause: Nodes were clearing the plan array
   - Solution: Removed plan clearing from individual nodes
   
2. **Intent Detection Order**: Fixed bug where pasted job descriptions were incorrectly classified
   - Root cause: Intent detection checked for "new_analysis" before "pasted_jd"
   - Solution: Reordered detection logic to check for pasted JDs first

3. **Error Handler Intervention**: Removed unwanted error messages appearing during normal operation
   - Root cause: Planner and executor nodes returning empty message arrays
   - Solution: Removed message keys from nodes that shouldn't generate user-facing messages

### Code Organization
- **Modularized Structure**: Moved all source files to `src/` directory
- **Consolidated Nodes**: All agent nodes are now in a single `src/nodes/nodes.py` file
- **Proper Imports**: Fixed all import paths to use absolute imports
- **Enhanced Logging**: Added comprehensive debug logging throughout the system

## Deployment

The application is configured for deployment using Docker and Railway:

### Docker Deployment
```bash
docker build -t learntube-ai .
docker run -p 8000:8000 --env-file .env learntube-ai
```

### Railway Deployment
The project includes `railway.toml` and `Dockerfile` for easy Railway deployment:
1. Connect your GitHub repository to Railway
2. Add environment variables in Railway dashboard
3. Deploy (Railway will automatically use the Dockerfile)

### Environment Variables for Production
```
GROQ_API_KEY=your_groq_api_key
APIFY_API_KEY=your_apify_api_key
SEARX_HOST=your_searx_instance_url
PORT=8000  # Railway sets this automatically
```

## API Endpoints

- `POST /start_analysis`: Initialize a new analysis session with a LinkedIn URL
- `POST /chat/{thread_id}`: Send a message in an existing session
- `GET /history/{thread_id}`: Retrieve conversation history
- `GET /`: Health check endpoint

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.