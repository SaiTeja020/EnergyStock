import os
import sys

# Ensure root directory is on the python path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

def main():
    import uvicorn
    # Import the FastAPI app instance from openenv.server.app
    uvicorn.run(
        "openenv.server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()
