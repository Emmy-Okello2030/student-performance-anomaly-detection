"""
Complete training script - trains all models and saves to database.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import joblib
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import PROCESSED_DATA_DIR, MODEL_DIR, RANDOM_SEED
from src.models.predictive.random_forest import RandomForestModel
from src.models.anomaly.isolation_forest import IsolationForestModel
from src.risk_engine.integrator import RiskIntegrator
from src.utils.database import DatabaseManager

def load_cleaned_data():
    """Load cleaned datasets."""
    data = {}
    
    uci_path = PROCESSED_DATA_DIR / 'uci_cleaned.csv'
    if uci_path.exists():
        df = pd.read_csv(uci_path)
        data['uci'] = df
        print(f"✅ Loaded UCI data: {df.shape}")
    
    return data

def prepare_uci_data(df):
    """Prepare UCI data for training."""
    # Target is 'passed' (created during cleaning)
    feature_cols = [col for col in df.columns if col not in 
                   ['passed', 'target_grade', 'target_passed', 'data_source']]
    
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['passed']
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    return X, y, feature_cols

def train_uci_models():
    """Train models on UCI dataset."""
    print("\n" + "="*80)
    print("TRAINING ON UCI DATASET")
    print("="*80)
    
    # Load data
    data = load_cleaned_data()
    if 'uci' not in data:
        print("❌ UCI data not found!")
        return None
    
    # Prepare data
    X, y, features = prepare_uci_data(data['uci'])
    print(f"\nFeatures: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train
    )
    
    print(f"\nTrain: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    
    # Train Random Forest
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train, X_val, y_val)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    rf_model.save_model('random_forest_uci.pkl')
    
    # Show feature importance
    importance = rf_model.get_feature_importance()
    if importance is not None:
        print("\nTop 5 Features:")
        print(importance.head())
    
    # Train Isolation Forest (unsupervised)
    if_model = IsolationForestModel()
    if_model.fit(X)
    if_model.save_model('isolation_forest_uci.pkl')
    
    # Get anomaly scores
    anomaly_scores = if_model.score_samples(X)
    
    # Save to database
    with DatabaseManager() as db:
        # Save model metadata
        db.cursor.execute('''
            INSERT INTO model_metadata (model_name, model_type, parameters, accuracy, file_path)
            VALUES (?, ?, ?, ?, ?)
        ''', ('Random Forest', 'classification', str(rf_model.params), 
              rf_metrics['accuracy'], str(MODEL_DIR / 'predictive/random_forest_uci.pkl')))
        
        # Save sample risk scores (first 100 students)
        risk_integrator = RiskIntegrator()
        
        for i in range(min(100, len(X_test))):
            if i < len(y_test):
                pred_proba = rf_model.predict_proba([X_test.iloc[i]])[0][1]
                anom_score = anomaly_scores[X_test.index[i]] if i < len(anomaly_scores) else 0.5
                
                # Normalize anomaly score (higher = more anomalous)
                anom_norm = 1 - (anom_score - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
                
                composite = risk_integrator.calculate_composite(
                    np.array([pred_proba]), 
                    np.array([anom_norm])
                )[0]
                
                risk_level, _ = risk_integrator.classify_risk(composite)
                
                db.insert_risk_score(
                    student_id=i,
                    pred_score=float(pred_proba),
                    anom_score=float(anom_norm),
                    composite=float(composite),
                    risk_level=risk_level,
                    confidence=0.85
                )
        
        print("\n✅ Saved risk scores to database")
    
    return rf_model, if_model

if __name__ == "__main__":
    rf, iforest = train_uci_models()
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)