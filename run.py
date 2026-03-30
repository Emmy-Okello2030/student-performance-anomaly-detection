"""
Master runner script - runs the complete system.
"""

import subprocess
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)

def main():
    """Run the complete system."""
    
    print_header("STUDENT PERFORMANCE ANOMALY DETECTION SYSTEM")
    print("1. Run data pipeline")
    print("2. Train models")
    print("3. Launch dashboard")
    print("4. Run all")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1' or choice == '4':
        print_header("STEP 1: DATA PIPELINE")
        subprocess.run([sys.executable, "src/data_pipeline/run_pipeline.py"])
    
    if choice == '2' or choice == '4':
        print_header("STEP 2: MODEL TRAINING")
        subprocess.run([sys.executable, "src/models/train_all.py"])
    
    if choice == '3' or choice == '4':
        print_header("STEP 3: LAUNCHING DASHBOARD")
        print("Dashboard will open at http://localhost:8501")
        subprocess.run(["streamlit", "run", "src/dashboard/app.py"])

if __name__ == "__main__":
    main()