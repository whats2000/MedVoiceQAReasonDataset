"""
Launch script for the MedVoiceQA Human Verification UI.

This script starts the Streamlit interface for reviewing processed samples.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit review interface."""
    
    # Get the path to the review interface
    ui_dir = Path(__file__).parent
    interface_path = ui_dir / "review_interface.py"
    
    if not interface_path.exists():
        print(f"Error: Review interface not found at {interface_path}")
        sys.exit(1)
    
    print("🏥 Starting MedVoiceQA Human Verification Interface...")
    print(f"Interface location: {interface_path}")
    print("Opening browser automatically...")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(interface_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error launching Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down interface...")
        sys.exit(0)


if __name__ == "__main__":
    main()
