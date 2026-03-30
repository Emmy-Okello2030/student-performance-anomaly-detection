"""
Isolation Forest anomaly detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from config import IF_PARAMS, MODEL_DIR

class IsolationForestModel:
    """Isolation Forest for anomaly detection."""
    
    def __init__(self, params=None):
        self.params = params or IF_PARAMS
        self.model = IsolationForest(**self.params)
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X):
        """Fit the model."""
        print("\n" + "="*60)
        print("FITTING ISOLATION FOREST")
        print("="*60)
        print(f"Data shape: {X.shape}")
        
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        
        self.model.fit(X)
        self.is_fitted = True
        
        # Analyze
        scores = self.model.score_samples(X)
        print(f"\nAnomaly score statistics:")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std:  {scores.std():.4f}")
        
        return self
    
    def predict(self, X):
        """Predict anomalies (1=normal, -1=anomaly)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.predict(X)
    
    def score_samples(self, X):
        """Get anomaly scores (lower = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.score_samples(X)
    
    def get_anomaly_flags(self, X, threshold_percentile=10):
        """Get anomaly flags based on percentile."""
        scores = self.score_samples(X)
        threshold = np.percentile(scores, threshold_percentile)
        flags = (scores <= threshold).astype(int)
        return flags, scores
    
    def save_model(self, filename='isolation_forest.pkl'):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        path = MODEL_DIR / 'anomaly' / filename
        joblib.dump({
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, path)
        print(f"\n✅ Model saved to {path}")
        return path
    
    def load_model(self, filename='isolation_forest.pkl'):
        """Load model from disk."""
        path = MODEL_DIR / 'anomaly' / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        data = joblib.load(path)
        self.model = data['model']
        self.params = data['params']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        
        print(f"\n✅ Model loaded from {path}")
        return self