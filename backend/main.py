import os
import sys

# Ensure project root is on sys.path
# In Docker, we are at /app/bess_rl/backend/main.py, so we want /app
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "bess_rl.openenv.server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info",
    )
