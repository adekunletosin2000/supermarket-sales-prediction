import shap
import numpy as np

def get_prediction_and_confidence(model, input_data):
    """
    Returns prediction and a realistic confidence score (0-100%).
    Confidence is estimated based on:
    - Model prediction stability using SHAP value magnitudes
    - Normalized between 50% and 95% for display purposes
    """
    # Predict
    pred = model.predict(input_data)[0]

    # --- SHAP-based confidence ---
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    # Confidence = 1 - normalized sum of absolute SHAP values
    abs_shap_sum = np.sum(np.abs(shap_values.values))
    max_possible = len(shap_values.values[0])  # max sum if each feature contributed 1
    raw_conf = 1 - (abs_shap_sum / max_possible)

    # Normalize to 50-95% for display
    confidence = max(0.5, min(raw_conf, 0.95))

    return pred, round(confidence * 100, 2)


def get_shap_values(model, input_data):
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    return shap_values