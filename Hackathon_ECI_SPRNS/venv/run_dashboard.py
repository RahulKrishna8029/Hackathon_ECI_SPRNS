#!/usr/bin/env python3
"""
Script to run the SPRNS Dashboard.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit dashboard."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, "Dashboard", "sprns_ui.py")
    
    # Check if the dashboard file exists
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    # Run Streamlit
    try:
        print("Starting SPRNS Dashboard...")
        print(f"Dashboard will be available at: http://localhost:8501")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
    except Exception as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()