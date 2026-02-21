import shap
import numpy as np
import pandas as pd

def get_prediction_and_confidence(model, input_data):
    pred_log = model.predict(input_data)
    pred_total = np.expm1(pred_log)[0]

    # Confidence proxy (distance from mean prediction)
    baseline = np.mean(model.predict(input_data))
    confidence = min(0.95, 0.6 + abs(pred_log[0] - baseline))

    return pred_total, round(confidence * 100, 2)


def get_shap_values(model, input_data):
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    return shap_values