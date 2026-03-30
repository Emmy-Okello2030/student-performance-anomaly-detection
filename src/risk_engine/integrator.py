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


class RiskExplainer:
    """
    Generates explanations for risk assessments.
    """
    
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