import os
import sys
import uvicorn
from pathlib import Path

if __name__ == "__main__":
    # Add project root to Python path so imports work
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Get PORT from environment, default to 8000
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting server on port {port}")
    
    # Run the FastAPI app
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )