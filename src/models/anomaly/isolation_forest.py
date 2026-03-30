"""
Isolation Forest anomaly detection.
"""

from sklearn.ensemble import IsolationForest
import joblib

class IsolationForestModel:
    def __init__(self, params=None):
        self.params = params or {'n_estimators': 100, 'contamination': 0.1}
        self.model = IsolationForest(**self.params)
        self.is_fitted = False
    
    def fit(self, X):
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
    
    def save_model(self, path):
        joblib.dump(self.model, path)