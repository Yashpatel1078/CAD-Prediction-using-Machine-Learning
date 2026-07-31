import numpy as np
import pandas as pd

def predict_risk_probability(model, X_instance):
    """
    Returns the CAD probability for a given feature matrix.
    """
    # Get positive class probability
    probabilities = model.predict_proba(X_instance)[:, 1]
    return probabilities

def predict_class_with_threshold(probabilities, threshold=0.2):
    """
    Applies custom probability threshold to classify patient.
    Optimizes for high recall (sensitive screening).
    """
    return (probabilities >= threshold).astype(int)

def format_prediction_output(probability, threshold=0.2):
    """
    Returns a human-readable prediction summary.
    """
    prediction = predict_class_with_threshold(np.array([probability]), threshold)[0]
    status = "At-Risk (High Probability of CAD)" if prediction == 1 else "Low-Risk"
    
    return {
        "status": status,
        "cad_probability": float(probability),
        "threshold_used": threshold,
        "risk_level": "High" if probability >= threshold else "Low"
    }
