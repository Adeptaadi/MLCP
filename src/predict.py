import joblib
import numpy as np

def load_models():

    models = {
        "logistic": joblib.load("models/logistic_model.pkl"),
        "svm": joblib.load("models/svm_model.pkl"),
        "rf": joblib.load("models/rf_model.pkl")
    }

    scaler = joblib.load("models/scaler_clf.pkl")

    return models, scaler


def predict(models, scaler, input_data):

    input_scaled = scaler.transform(input_data)

    results = {}

    for name, model in models.items():
        pred = model.predict(input_scaled)[0]

        try:
            prob = model.predict_proba(input_scaled)[0][1]
        except:
            prob = None

        results[name] = {
            "prediction": int(pred),
            "probability": float(prob) if prob else None
        }

    return results