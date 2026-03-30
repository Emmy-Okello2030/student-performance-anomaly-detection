"""
Master configuration file for Student Performance Anomaly Detection System.
All settings in one place.
"""
import os
from pathlib import Path

# =========================================================
# PATHS - Change these if you move your project
# =========================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'
LOG_DIR = OUTPUT_DIR / 'logs'

# Create directories if they don't exist
for dir_path in [PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =========================================================
# DATASET PATHS
# =========================================================
OULAD_PATH = RAW_DATA_DIR / 'oulad'
UCI_PATH = RAW_DATA_DIR / 'uci'

# =========================================================
# MODEL PARAMETERS
# =========================================================
RANDOM_SEED = 42

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': RANDOM_SEED,
    'class_weight': 'balanced',
    'n_jobs': -1
}

# XGBoost parameters
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'eval_metric': 'logloss'
}

# Isolation Forest parameters
IF_PARAMS = {
    'n_estimators': 100,
    'contamination': 0.1,
    'random_state': RANDOM_SEED
}

# DBSCAN parameters
DBSCAN_PARAMS = {
    'eps': 0.5,
    'min_samples': 5,
    'metric': 'euclidean'
}

# =========================================================
# RISK SCORING SETTINGS
# =========================================================
PREDICTIVE_WEIGHT = 0.6
ANOMALY_WEIGHT = 0.4

RISK_THRESHOLDS = {
    'LOW': 0.3,
    'MEDIUM': 0.6,
    'HIGH': 0.8,
    'CRITICAL': 1.0
}

# =========================================================
# DATABASE SETTINGS
# =========================================================
DATABASE_PATH = PROCESSED_DATA_DIR / 'anomaly_detection.db'
DATABASE_TIMEOUT = 30

# =========================================================
# LOGGING SETTINGS
# =========================================================
LOG_LEVEL = 'INFO'
LOG_FILE = LOG_DIR / 'app.log'