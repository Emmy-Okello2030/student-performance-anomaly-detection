"""
Main pipeline runner - orchestrates the complete data cleaning process.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_pipeline.complete_cleaner import CompleteDataCleaner

def run_pipeline():
    """Run the complete data pipeline."""
    
    print("\n" + "="*80)
    print("STUDENT PERFORMANCE ANOMALY DETECTION SYSTEM")
    print("COMPLETE DATA PIPELINE")
    print("="*80)
    
    # Clean data
    print("\n📥 Cleaning datasets...")
    cleaner = CompleteDataCleaner(base_path='data/raw')
    cleaned_data = cleaner.run_complete_cleaning()
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return cleaned_data

if __name__ == "__main__":
    run_pipeline()