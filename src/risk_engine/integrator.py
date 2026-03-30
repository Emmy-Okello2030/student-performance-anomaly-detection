
"""
Risk integration engine.
"""

class RiskIntegrator:
    def __init__(self, pred_weight=0.6, anom_weight=0.4):
        self.pred_weight = pred_weight
        self.anom_weight = anom_weight
    
    def calculate_composite(self, pred_scores, anom_scores):
        # Simple weighted average
        return (self.pred_weight * pred_scores) + (self.anom_weight * anom_scores)