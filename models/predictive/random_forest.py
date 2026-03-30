"""
Random Forest model implementation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from config import RF_PARAMS, MODEL_DIR, RANDOM_SEED

class RandomForestModel:
    """Random Forest classifier for student performance prediction."""
    
    def __init__(self, params=None):
        self.params = params or RF_PARAMS
        self.model = RandomForestClassifier(**self.params)
        self.feature_names = None
        self.is_trained = False
        self.training_history = {}
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        print(f"Training data: {X_train.shape}")
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        
        # Train
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"\nTraining accuracy: {train_acc:.4f}")
        self.training_history['train_accuracy'] = train_acc
        
        # Validation if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Validation accuracy: {val_acc:.4f}")
            self.training_history['val_accuracy'] = val_acc
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"\n5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        self.training_history['cv_mean'] = cv_scores.mean()
        
        return self
    
    def predict(self, X):
        """Generate predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Generate probability scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return metrics
    
    def get_feature_importance(self, top_n=10):
        """Get top N feature importances."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            return importance_df
        return importances
    
    def save_model(self, filename='random_forest.pkl'):
        """Save model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        path = MODEL_DIR / 'predictive' / filename
        joblib.dump({
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }, path)
        print(f"\n✅ Model saved to {path}")
        return path
    
    def load_model(self, filename='random_forest.pkl'):
        """Load model from disk."""
        path = MODEL_DIR / 'predictive' / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        data = joblib.load(path)
        self.model = data['model']
        self.params = data['params']
        self.feature_names = data['feature_names']
        self.is_trained = data['is_trained']
        self.training_history = data.get('training_history', {})
        
        print(f"\n✅ Model loaded from {path}")
        return self