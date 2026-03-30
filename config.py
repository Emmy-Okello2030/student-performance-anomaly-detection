"""
Configuration file for Student Performance Anomaly Detection System.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'
DATABASE_PATH = PROCESSED_DATA_DIR / 'anomaly_detection.db'


# Model parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

# Risk thresholds
RISK_THRESHOLDS = {
    'LOW': 0.3,
    'MEDIUM': 0.6,
    'HIGH': 0.8,
    'CRITICAL': 1.0
}