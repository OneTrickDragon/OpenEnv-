"""
server/app.py — FastAPI server using open_env.core.

create_app() wires up /reset, /step, /state, /health, /ws, /web.
GET / added so HuggingFace Space health probes return 200.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_app

from models import DataCleaningAction, DataCleaningObservation
from server.dc_environment import DataCleaningEnvironment

app = create_app(DataCleaningEnvironment, DataCleaningAction, DataCleaningObservation)


@app.get("/")
def root():
    return {
        "name":    "Data Cleaning OpenEnv",
        "version": "0.1.0",
        "status":  "running",
        "tasks":   ["ecommerce_easy", "patient_records_medium", "financial_audit_hard"],
        "endpoints": {
            "health":    "/health",
            "reset":     "POST /reset",
            "step":      "POST /step",
            "state":     "GET  /state",
            "websocket": "/ws",
            "docs":      "/docs",
        },
    }


def main():
    """Entrypoint for `uv run server` and direct execution."""
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()