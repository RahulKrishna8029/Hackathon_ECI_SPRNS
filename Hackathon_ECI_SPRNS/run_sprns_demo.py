#!/usr/bin/env python3
"""
Launch script for SPRNS Demo with Mock Graph Data.
This script runs the demo UI with the 4-customer mock graph.
"""

import subprocess
import sys
import os

def main():
    """Run the SPRNS demo with mock graph data."""
    
    # Get paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_path = os.path.join(project_root, "sprns_env")
    demo_ui_path = os.path.join(project_root, "venv", "Dashboard", "demo_ui.py")
    
    # Check if virtual environment exists
    if not os.path.exists(venv_path):
        print("❌ Virtual environment not found. Please run the setup first.")
        sys.exit(1)
    
    # Check if demo UI exists
    if not os.path.exists(demo_ui_path):
        print("❌ Demo UI file not found.")
        sys.exit(1)
    
    print("🚀 Starting SPRNS Demo with Mock Customer Graph...")
    print("📊 Mock Database includes:")
    print("   • 4 Customers: TechCorp, HealthFirst, GreenEnergy, RetailMax")
    print("   • 3 Products: Analytics Platform, Security Suite, IoT System")
    print("   • 6 Documents: Case studies, reports, implementation guides")
    print("   • Multiple customer relationships and interactions")
    print()
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Activate virtual environment and run streamlit
    try:
        # Use the virtual environment's python and streamlit
        venv_python = os.path.join(venv_path, "bin", "python")
        venv_streamlit = os.path.join(venv_path, "bin", "streamlit")
        
        subprocess.run([
            venv_streamlit, "run", demo_ui_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down SPRNS Demo...")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()