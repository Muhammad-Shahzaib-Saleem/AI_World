#!/usr/bin/env python3
"""
Run script for Email RAG API
Provides options to run backend, frontend, or both
"""

import os
import sys
import subprocess
import argparse
import signal
import time
from pathlib import Path

def run_backend(host="0.0.0.0", port=8000, reload=True):
    """Run FastAPI backend"""
    print(f"🚀 Starting FastAPI backend on {host}:{port}")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    return subprocess.Popen(cmd)

def run_frontend(host="0.0.0.0", port=8501):
    """Run Streamlit frontend"""
    print(f"🖥️ Starting Streamlit frontend on {host}:{port}")
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "frontend.py",
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true"
    ]
    
    return subprocess.Popen(cmd)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "streamlit",
        "langchain",
        "openai"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def setup_environment():
    """Setup environment variables and configuration"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from .env.example")
        env_file.write_text(env_example.read_text())
        print("⚠️  Please edit .env file with your configuration")
    
    # Load environment variables
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")

def signal_handler(sig, frame, processes):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down services...")
    for process in processes:
        if process and process.poll() is None:
            process.terminate()
    
    # Wait for processes to terminate
    time.sleep(2)
    
    for process in processes:
        if process and process.poll() is None:
            process.kill()
    
    print("✅ All services stopped")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Email RAG API Runner")
    parser.add_argument(
        "mode",
        choices=["backend", "frontend", "both"],
        help="Which service to run"
    )
    parser.add_argument(
        "--backend-host",
        default="0.0.0.0",
        help="Backend host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8000,
        help="Backend port (default: 8000)"
    )
    parser.add_argument(
        "--frontend-host",
        default="0.0.0.0",
        help="Frontend host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=8501,
        help="Frontend port (default: 8501)"
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload for backend"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps or not check_dependencies():
        return
    
    # Setup environment
    setup_environment()
    
    processes = []
    
    try:
        if args.mode in ["backend", "both"]:
            backend_process = run_backend(
                host=args.backend_host,
                port=args.backend_port,
                reload=not args.no_reload
            )
            processes.append(backend_process)
            
            if args.mode == "both":
                # Wait a bit for backend to start
                time.sleep(3)
        
        if args.mode in ["frontend", "both"]:
            # Set API URL for frontend
            api_url = f"http://{args.backend_host}:{args.backend_port}"
            os.environ["API_BASE_URL"] = api_url
            
            frontend_process = run_frontend(
                host=args.frontend_host,
                port=args.frontend_port
            )
            processes.append(frontend_process)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, processes))
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, processes))
        
        print("\n🎉 Services started successfully!")
        
        if args.mode == "backend":
            print(f"📚 API Documentation: http://{args.backend_host}:{args.backend_port}/docs")
            print(f"📖 ReDoc Documentation: http://{args.backend_host}:{args.backend_port}/redoc")
        elif args.mode == "frontend":
            print(f"🖥️ Frontend Interface: http://{args.frontend_host}:{args.frontend_port}")
        else:  # both
            print(f"📚 API Documentation: http://{args.backend_host}:{args.backend_port}/docs")
            print(f"🖥️ Frontend Interface: http://{args.frontend_host}:{args.frontend_port}")
        
        print("\n💡 Press Ctrl+C to stop all services")
        
        # Wait for processes
        while True:
            for process in processes:
                if process.poll() is not None:
                    print(f"❌ Process {process.pid} has stopped unexpectedly")
                    signal_handler(None, None, processes)
            time.sleep(1)
    
    except KeyboardInterrupt:
        signal_handler(None, None, processes)
    except Exception as e:
        print(f"❌ Error: {e}")
        signal_handler(None, None, processes)

if __name__ == "__main__":
    main()