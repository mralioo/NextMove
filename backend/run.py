"""
Run the FastAPI backend.
    python backend/run.py
"""
import sys, os
from pathlib import Path
import uvicorn

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")

if __name__ == '__main__':
    uvicorn.run('backend.api:app', host='0.0.0.0', port=8000, reload=True, app_dir=PROJECT_ROOT)
