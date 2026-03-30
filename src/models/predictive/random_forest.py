"""
Random Forest model implementation.
"""

from sklearn.ensemble import RandomForestClassifier
import joblib

class RandomForestModel:
    def __init__(self, params=None):
        self.params = params or {'n_estimators': 100, 'max_depth': 10}
        self.model = RandomForestClassifier(**self.params)
        self.is_trained = False
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def save_model(self, path):
        joblib.dump(self.model, path)
        