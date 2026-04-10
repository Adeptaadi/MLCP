# =========================
# IMPORTS
# =========================
import shap
import numpy as np

# =========================
# CLASSIFICATION (RF ONLY)
# =========================
def explain_classification(models, input_data, background_data):
    """
    Explain classification models (uses SHAP for Random Forest only)
    """

    explanations = {}

    for name, model in models.items():

        # ✅ Use SHAP only for Random Forest
        if name == "rf":
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)

                explanations[name] = {
                    "type": "shap",
                    "explainer": explainer,
                    "shap_values": shap_values
                }

            except Exception as e:
                explanations[name] = {"error": str(e)}

        # ❌ Skip SVM (not SHAP-friendly)
        elif name == "svm":
            explanations[name] = {
                "type": "info",
                "message": "SHAP not supported efficiently for SVM"
            }

        # ✔ Logistic Regression → coefficients
        elif name == "logistic":
            try:
                explanations[name] = {
                    "type": "coefficients",
                    "values": model.coef_[0]
                }
            except:
                explanations[name] = {"error": "Could not extract coefficients"}

    return explanations


# =========================
# REGRESSION (RF ONLY)
# =========================
def explain_regression(models, input_data, background_data):
    """
    Explain regression models (uses SHAP for Random Forest Regressor)
    """

    explanations = {}

    for name, model in models.items():

        # ✅ SHAP for RF regressor
        if name == "rf_reg":
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)

                explanations[name] = {
                    "type": "shap",
                    "explainer": explainer,
                    "shap_values": shap_values
                }

            except Exception as e:
                explanations[name] = {"error": str(e)}

        # Linear models → coefficients
        elif name in ["linear", "ridge", "lasso"]:
            try:
                explanations[name] = {
                    "type": "coefficients",
                    "values": model.coef_
                }
            except:
                explanations[name] = {"error": "Could not extract coefficients"}

    return explanations


# =========================
# MAIN EXPLAIN FUNCTION
# =========================
def explain(mode, models, input_data, background_data=None):
    """
    Wrapper function for explainability
    mode: "classification" or "regression"
    """

    if mode == "classification":
        return explain_classification(models, input_data, background_data)

    elif mode == "regression":
        return explain_regression(models, input_data, background_data)

    else:
        raise ValueError("Invalid mode. Choose 'classification' or 'regression'")


# =========================
# TEST (OPTIONAL)
# =========================
if __name__ == "__main__":

    import joblib
    import numpy as np

    print("Testing explain module...")

    # Dummy test input
    sample = np.random.rand(1, 20)

    try:
        rf_model = joblib.load("models/rf_model.pkl")

        models = {
            "rf": rf_model
        }

        explanations = explain("classification", models, sample)

        print("Explanation generated successfully!")
        print(explanations)

    except Exception as e:
        print("Error:", e)