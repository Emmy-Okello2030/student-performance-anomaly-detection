"""
Risk integration engine - combines predictive and anomaly scores.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import PREDICTIVE_WEIGHT, ANOMALY_WEIGHT, RISK_THRESHOLDS

class RiskIntegrator:
    """Combines predictive and anomaly detection scores."""
    
    def __init__(self, pred_weight=PREDICTIVE_WEIGHT, anom_weight=ANOMALY_WEIGHT):
        self.pred_weight = pred_weight
        self.anom_weight = anom_weight
        self.thresholds = RISK_THRESHOLDS
    
    def calculate_composite(self, pred_scores, anom_scores):
        """Calculate composite risk score."""
        # Normalize to 0-1
        pred_norm = (pred_scores - pred_scores.min()) / (pred_scores.max() - pred_scores.min() + 1e-8)
        anom_norm = (anom_scores - anom_scores.min()) / (anom_scores.max() - anom_scores.min() + 1e-8)
        
        # Weighted combination
        composite = (self.pred_weight * pred_norm) + (self.anom_weight * anom_norm)
        return composite
    
    def classify_risk(self, composite_score):
        """Classify risk level."""
        if composite_score < self.thresholds['LOW']:
            return 'LOW', '🟢'
        elif composite_score < self.thresholds['MEDIUM']:
            return 'MEDIUM', '🟡'
        elif composite_score < self.thresholds['HIGH']:
            return 'HIGH', '🟠'
        else:
            return 'CRITICAL', '🔴'
    
    def get_risk_color(self, risk_level):
        """Get color for risk level."""
        colors = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🟠',
            'CRITICAL': '🔴'
        }
        return colors.get(risk_level, '⚪')

class RiskExplainer:
    """Generates explanations for risk assessments."""
    
    def __init__(self):
        self.templates = {
            'login_drop': "Login frequency dropped {drop:.0f}% below personal average",
            'missed_assignments': "Missed {count} consecutive assignments",
            'submission_delay': "Submission delay of {delay:.1f} days vs class average",
            'low_performance': "Performance below peer average (percentile: {pct:.0f})",
            'high_anomaly': "Unusual behavioral pattern detected"
        }
    
    def generate_explanation(self, factors):
        """Generate explanation from risk factors."""
        if not factors:
            return "No significant risk factors identified."
        
        explanation = "This student was flagged because:\n"
        for factor in factors[:3]:
            explanation += f"• {factor}\n"
        
        return explanation
    
    def get_recommendations(self, risk_level):
        """Get recommendations based on risk level."""
        recs = {
            'LOW': "Continue normal monitoring.",
            'MEDIUM': "Monitor closely. Consider sending encouragement email.",
            'HIGH': "Contact student. Schedule check-in meeting.",
            'CRITICAL': "IMMEDIATE ACTION: Schedule one-on-one meeting with academic advisor."
        }
        return recs.get(risk_level, "Review student record.")