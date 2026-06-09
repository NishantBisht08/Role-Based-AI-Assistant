import subprocess
import time
import sys
import os

def main():
    print("==========================================")
    print("Starting FinSolve Enterprise AI Services")
    print("==========================================")
    
    # Start FastAPI backend
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    print("Backend starting on port 8000...")
    # Wait a moment for backend to initialize
    time.sleep(3)
    
    # Start Streamlit frontend
    print("Frontend starting on port 8501...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/frontend_app.py", "--server.port", "8501", "--server.address", "127.0.0.1"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    print("\n[SUCCESS] Services Running!")
    print("UI Address: http://127.0.0.1:8501")
    print("Press Ctrl+C to stop both services.\n")
    
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
