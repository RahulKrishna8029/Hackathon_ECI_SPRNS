#!/usr/bin/env python3
"""
Script to run the SPRNS Demo Dashboard.
Lightweight version that works without heavy ML dependencies.
"""

import subprocess
import sys
import os

def check_streamlit():
    """Check if Streamlit is available."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def main():
    """Run the demo dashboard."""
    # Check if Streamlit is available
    if not check_streamlit():
        print("❌ Streamlit is not installed.")
        print("Please install it with: pip install streamlit")
        sys.exit(1)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, "Dashboard", "demo_ui.py")
    
    # Check if the dashboard file exists
    if not os.path.exists(dashboard_path):
        print(f"Error: Demo dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    # Run Streamlit
    try:
        print("🚀 Starting SPRNS Demo Dashboard...")
        print("📊 Dashboard will be available at: http://localhost:8501")
        print("💡 This is a lightweight demo that simulates RAG functionality")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down demo dashboard...")
    except Exception as e:
        print(f"❌ Error running demo dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()