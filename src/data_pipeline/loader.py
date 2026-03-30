"""
Data loading module. Handles loading and initial validation of datasets.
From SDS Section 3.1.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import OULAD_PATH, UCI_PATH

class DataLoader:
    """
    Loads and validates datasets.
    SDS Section 3.1.1 - DataLoader Class
    """
    
    def __init__(self):
        self.oulad_path = OULAD_PATH
        self.uci_path = UCI_PATH
        self.loaded_files = {}
        
    def load_oulad(self):
        """
        Load all OULAD CSV files into a dictionary of DataFrames.
        Returns: Dict[str, pd.DataFrame]
        """
        print("\n" + "="*60)
        print("Loading OULAD dataset...")
        print("="*60)
        
        files = {
            'student_info': 'studentInfo.csv',
            'student_reg': 'studentRegistration.csv',
            'student_vle': 'studentVle.csv',
            'vle': 'vle.csv',
            'assessments': 'assessments.csv',
            'student_assessment': 'studentAssessment.csv'
        }
        
        data = {}
        for key, filename in files.items():
            filepath = self.oulad_path / filename
            if filepath.exists():
                print(f"  Loading {filename}...")
                # Use low_memory=False to avoid dtype warnings
                data[key] = pd.read_csv(filepath, low_memory=False)
                print(f"    Shape: {data[key].shape}")
                print(f"    Columns: {list(data[key].columns)}")
            else:
                print(f"  WARNING: {filename} not found at {filepath}")
        
        self.loaded_files['oulad'] = data
        return data
    
    def load_uci(self):
        """
        Load UCI dataset.
        Returns: Dict[str, pd.DataFrame]
        """
        print("\n" + "="*60)
        print("Loading UCI dataset...")
        print("="*60)
        
        data = {}
        files = ['student-mat.csv', 'student-por.csv']
        
        for filename in files:
            filepath = self.uci_path / filename
            if filepath.exists():
                print(f"  Loading {filename}...")
                # UCI files use semicolon delimiter
                df = pd.read_csv(filepath, sep=';')
                key = filename.replace('.csv', '')
                data[key] = df
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
        
        self.loaded_files['uci'] = data
        return data
    
    def validate_schema(self, df, name="", expected_columns=None):
        """
        Validate DataFrame against expected schema.
        """
        print(f"\nValidating {name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Missing values: {df.isnull().sum().sum()}")
        print(f"  Duplicates: {df.duplicated().sum()}")
        print(f"  Data types:\n{df.dtypes.value_counts()}")
        
        return {
            'shape': df.shape,
            'missing': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum()
        }

# Quick test
if __name__ == "__main__":
    loader = DataLoader()
    oulad_data = loader.load_oulad()
    uci_data = loader.load_uci()
    
    if 'student_info' in oulad_data:
        loader.validate_schema(oulad_data['student_info'], "OULAD Student Info")