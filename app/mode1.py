import streamlit as st
import numpy as np
import joblib
import shap
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict

def generate_realistic_clf(features):

    values = {}

    # 🔥 Core numeric features (realistic distributions)

    values["age"] = int(np.clip(np.random.normal(55, 10), 29, 77))
    values["sex"] = np.random.choice([0, 1], p=[0.3, 0.7])  # more males

    values["trestbps"] = int(np.clip(np.random.normal(130, 20), 90, 200))
    values["chol"] = int(np.clip(np.random.normal(240, 50), 100, 600))

    values["thalach"] = int(np.clip(np.random.normal(150, 25), 70, 210))
    values["oldpeak"] = round(np.clip(np.random.exponential(1.0), 0, 6), 2)

    values["fbs"] = np.random.choice([0, 1], p=[0.85, 0.15])
    values["exang"] = np.random.choice([0, 1], p=[0.7, 0.3])

    values["ca"] = np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.2, 0.1, 0.07, 0.03])

    # 🔥 Initialize all features to 0
    final = {f: 0 for f in features}

    # Fill numeric
    for f in features:
        if f in values:
            final[f] = values[f]

    # 🔥 One-hot categorical groups

    def assign_one_hot(prefix, choices):
        group = [f for f in features if prefix in f]
        if not group:
            return

        selected = np.random.choice(choices)

        for f in group:
            if f.endswith(str(selected)):
                final[f] = 1
            else:
                final[f] = 0

    assign_one_hot("cp_", [0, 1, 2, 3])
    assign_one_hot("restecg_", [0, 1, 2])
    assign_one_hot("slope_", [0, 1, 2])
    assign_one_hot("thal_", [0, 1, 2, 3])

    # 🔥 Convert to array (ordered)
    return np.array([final[f] for f in features])


def regression_inputs(features):

    inputs = {}

    col1, col2 = st.columns(2)

    for i, f in enumerate(features):

        col = col1 if i % 2 == 0 else col2

        with col:

            if f == "longitude":
                inputs[f] = st.slider(f, -124.5, -114.0, -120.0)

            elif f == "latitude":
                inputs[f] = st.slider(f, 32.0, 42.0, 35.0)

            elif f == "housing_median_age":
                inputs[f] = st.slider(f, 1, 52, 25)

            elif f == "total_rooms":
                inputs[f] = st.slider(f, 10, 20000, 5000)

            elif f == "total_bedrooms":
                inputs[f] = st.slider(f, 1, 6000, 1000)

            elif f == "population":
                inputs[f] = st.slider(f, 10, 25000, 3000)

            elif f == "households":
                inputs[f] = st.slider(f, 1, 6000, 1000)

            elif f == "median_income":
                inputs[f] = st.slider(f, 0.5, 15.0, 5.0)

    return inputs


def ocean_input():

    ocean_type = st.selectbox(
        "Ocean Proximity",
        ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
    )

    ocean_features = {
        "ocean_proximity_INLAND": 0,
        "ocean_proximity_NEAR BAY": 0,
        "ocean_proximity_NEAR OCEAN": 0,
        "ocean_proximity_ISLAND": 0
    }

    key = f"ocean_proximity_{ocean_type}"
    ocean_features[key] = 1

    return ocean_features

def generate_realistic_input_reg(features):

    values = {}

    values["longitude"] = np.random.uniform(-124, -114)
    values["latitude"] = np.random.uniform(32, 42)
    values["housing_median_age"] = np.random.randint(1, 52)
    values["total_rooms"] = np.random.randint(100, 20000)
    values["total_bedrooms"] = np.random.randint(50, 6000)
    values["population"] = np.random.randint(100, 25000)
    values["households"] = np.random.randint(50, 6000)
    values["median_income"] = np.random.uniform(1, 15)

    ocean_types = ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
    selected = np.random.choice(ocean_types)

    for o in ocean_types:
        values[f"ocean_proximity_{o}"] = 1 if o == selected else 0

    return values
def run_mode1():

    tab = st.tabs(["Classification", "Regression"])

    with tab[0]:

        st.header("🧠 Heart Disease Prediction")

        models = {
            "Logistic": joblib.load("models/logistic_model.pkl"),
            "SVM": joblib.load("models/svm_model.pkl"),
            "RF": joblib.load("models/rf_model.pkl")
        }

        scaler = joblib.load("models/scaler_clf.pkl")
        features = joblib.load("models/feature_names.pkl")

        if st.button("🎲 Random Input (Clf)"):
            st.session_state["clf"] = generate_realistic_clf(features)

        if "clf" not in st.session_state:
            st.session_state["clf"] = np.zeros(len(features))

        vals = st.session_state["clf"]

        cols = st.columns(3)
        for i, f in enumerate(features):
            vals[i] = cols[i % 3].number_input(f, value=float(vals[i]), key=f"clf_{i}")

        X = np.array(vals).reshape(1, -1)

        if st.button("🚀 Predict (Clf)"):

            res = predict(models, scaler, X)

            st.subheader("Model Comparison")

            cols = st.columns(3)
            for i, (name, r) in enumerate(res.items()):
                with cols[i]:
                    st.metric(name, f"Class {r['prediction']}", f"{r['probability']:.2f}")

            # SHAP
            st.subheader("SHAP (RF)")

            explainer = shap.TreeExplainer(models["RF"])
            shap_vals = explainer.shap_values(X)

            df = pd.DataFrame({
                "Feature": features,
                "Impact": shap_vals[1][0]
            }).sort_values(by="Impact", key=abs, ascending=False)

            st.bar_chart(df.set_index("Feature"))
    

    with tab[1]:

        st.header("🏠 House Price Prediction")

        models = {
            "Linear": joblib.load("models/linear_model.pkl"),
            "Ridge": joblib.load("models/ridge_model.pkl"),
            "Lasso": joblib.load("models/lasso_model.pkl"),
            "RF": joblib.load("models/rf_reg_model.pkl")
        }

        scaler = joblib.load("models/scaler_reg.pkl")
        features = joblib.load("models/feature_names_reg.pkl")

        # 🎲 RANDOM
        if st.button("🎲 Random Input (Reg)"):
            st.session_state["reg"] = generate_realistic_input_reg(features)

        if "reg" not in st.session_state:
            st.session_state["reg"] = generate_realistic_input_reg(features)

        # 🔥 UI INPUTS
        numeric_inputs = regression_inputs(features)
        ocean_features = ocean_input()

        # Merge
        final_input = {**numeric_inputs, **ocean_features}

        # Convert to ordered array
        X = np.array([final_input[f] for f in features]).reshape(1, -1)
        X_scaled = scaler.transform(X)

        if st.button("🚀 Predict (Reg)"):

            st.subheader("Model Comparison")

            cols = st.columns(4)
            for i, (name, model) in enumerate(models.items()):
                pred = model.predict(X_scaled)[0]
                with cols[i]:
                    st.metric(name, f"{pred:,.2f}")

            # SHAP
            st.subheader("SHAP (RF Regression)")

            explainer = shap.TreeExplainer(models["RF"])
            shap_vals = explainer.shap_values(X_scaled)

            df = pd.DataFrame({
                "Feature": features,
                "Impact": shap_vals[0]
            }).sort_values(by="Impact", key=abs, ascending=False)

            st.bar_chart(df.set_index("Feature"))