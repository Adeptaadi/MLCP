# import streamlit as st
# import pandas as pd
# import numpy as np
# import shap
# import warnings
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# warnings.filterwarnings("ignore")

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Models
# from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# # Metrics
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     mean_absolute_error, r2_score
# )


# def run_mode2():

#     st.title("📊 Mode 2: Dataset Analysis & Explainability")

#     # =========================
#     # FILE UPLOAD
#     # =========================
#     file = st.file_uploader("Upload CSV Dataset", type=["csv"])

#     if file is None:
#         st.info("Please upload a dataset to begin")
#         return

#     # =========================
#     # LOAD DATA
#     # =========================
#     df = pd.read_csv(file)

#     st.subheader("📄 Dataset Preview")
#     st.dataframe(df.head())

#     # =========================
#     # DATASET ANALYSIS
#     # =========================
#     st.subheader("📊 Dataset Diagnostics")

#     st.write("Shape:", df.shape)

#     st.write("Missing Values per Column:")
#     st.write(df.isnull().sum())

#     st.write("Data Types:")
#     st.write(df.dtypes)

#     # =========================
#     # CLEANING
#     # =========================
#     df.replace("?", np.nan, inplace=True)
#     df.dropna(inplace=True)

#     # =========================
#     # TARGET
#     # =========================
#     st.subheader("🎯 Target Selection")

#     target = st.selectbox("Select Target Column", df.columns)

#     X = df.drop(target, axis=1)
#     y = df[target]

#     # =========================
#     # ENCODING
#     # =========================
#     X = pd.get_dummies(X, drop_first=True)

#     # =========================
#     # TASK DETECTION
#     # =========================
#     num_classes = y.nunique()
#     if y.dtype == "object" or num_classes <= 10:
#         task = "classification"
#     else:
#         task = "regression"

#     if task == "classification":
#         if num_classes == 2:
#             st.info("Binary Classification detected")
#         else:
#             st.info(f"Multiclass Classification detected ({num_classes} classes)")
#     st.success(f"Detected Task: {task.upper()}")

#     # =========================
#     # SPLIT
#     # =========================
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # =========================
#     # SCALING
#     # =========================
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # =========================
#     # PREPROCESSING REPORT
#     # =========================
#     st.subheader("⚙️ Preprocessing Pipeline")

#     st.markdown("""
#     - Missing values removed
#     - Categorical features encoded using One-Hot Encoding
#     - Features scaled using StandardScaler
#     - Data split into training (80%) and testing (20%)
#     """)

#     # =========================
#     # MODEL SELECTION
#     # =========================
#     if task == "classification":
#         models = {
#             "Logistic": LogisticRegression(max_iter=1000),
#             "SVM": SVC(probability=True),
#             "Random Forest": RandomForestClassifier()
#         }
#     else:
#         models = {
#             "Linear": LinearRegression(),
#             "Ridge": Ridge(),
#             "Lasso": Lasso(),
#             "Random Forest": RandomForestRegressor()
#         }

#     # =========================
#     # TRAINING LOGS
#     # =========================
#     st.subheader("🤖 Model Training")

#     results = {}
#     predictions={}
#     for name, model in models.items():

#         st.write(f"Training {name}...")

#         model.fit(X_train, y_train)
#         pred = model.predict(X_test)
#         predictions[name]=pred

#         if task == "classification":
#             if num_classes == 2:
#                 avg = "binary"
#             else:
#                 avg = "weighted"
#             results[name] = {
#                     "Accuracy": accuracy_score(y_test, pred),
#                     "Precision": precision_score(y_test, pred, average=avg, zero_division=0),
#                     "Recall": recall_score(y_test, pred, average=avg, zero_division=0),
#                     "F1 Score": f1_score(y_test, pred, average=avg, zero_division=0)
#                 }
#         else:
#             results[name] = {
#                 "MAE": mean_absolute_error(y_test, pred),
#                 "R2": r2_score(y_test, pred)
#             }

#     # =========================
#     # RESULTS TABLE
#     # =========================
#     res_df = pd.DataFrame(results).T
#     for col in res_df.columns:
#         res_df[col] = pd.to_numeric(res_df[col])
#     st.subheader("📈 Model Performance Comparison")
#     st.dataframe(res_df.astype(float))

#     # =========================
#     # METRIC EXPLANATION
#     # =========================
#     st.subheader("📊 Metric Interpretation")

#     if task == "classification":
#         if task == "classification":

#             st.subheader("📉 Confusion Matrix")
#             show_norm = st.checkbox("Show Normalized Confusion Matrix")
#             labels = sorted(y.unique())
#             cm = confusion_matrix(y_test, best_pred, labels=labels)
#             if show_norm:
#                 with np.errstate(all='ignore'):  # prevent division warnings
#                     cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
#                     cm_display = np.nan_to_num(cm_display)  # handle division by zero
#                 fmt = ".2f"
#                 cmap = "Greens"
#                 title = "Normalized Confusion Matrix"
#             else:
#                 cm_display = cm
#                 fmt = "d"
#                 cmap = "Blues"
#                 title = "Confusion Matrix (Counts)"
#             fig, ax = plt.subplots(figsize=(10, 8))
#             sns.heatmap(
#                 cm_display,
#                 annot=True,
#                 fmt=fmt,
#                 cmap=cmap,
#                 xticklabels=labels,
#                 yticklabels=labels,
#                 cbar=True,
#                 ax=ax
#             )

#             ax.set_xlabel("Predicted Label")
#             ax.set_ylabel("True Label")
#             ax.set_title(title)

#             # Rotate labels
#             plt.xticks(rotation=45, ha="right")
#             plt.yticks(rotation=0)

#             # Display
#             st.pyplot(fig, use_container_width=True)

#         if num_classes == 2:
#             st.write("""
#             Binary Classification Metrics:
#             - Accuracy: Overall correctness
#             - Precision: Correct positive predictions
#             - Recall: Ability to detect positives
#             - F1 Score: Balance between precision and recall
#             """)
#         else:
#             st.write("""
#             Multiclass Classification Metrics:
#             - Accuracy: Overall correctness
#             - Precision/Recall/F1: Computed using weighted average
#             - Weighted average accounts for class imbalance
#             """)
#     else:
#         st.write("""
#         - MAE: Average prediction error  
#         - R²: Proportion of variance explained by model  
#         """)

#     # =========================
#     # BEST MODEL
#     # =========================
#     if task == "classification":
#         best = res_df["Accuracy"].idxmax()
#         best_pred = predictions[best]
#         best_score = res_df.loc[best, "Accuracy"]
#     else:
#         best = res_df["R2"].idxmax()
#         best_score = res_df.loc[best, "R2"]

#     st.success(f"🏆 Best Model: {best}")
#     st.subheader("📊 Target Distribution")

#     st.write(y.value_counts())
#     # =========================
#     # JUSTIFICATION
#     # =========================
#     st.subheader("🧠 Model Selection Justification")

#     if task == "classification":
#         st.write(f"""
#         {best} achieved the highest accuracy of {best_score:.2f},
#         indicating superior classification performance and generalization.
#         """)
#     else:
#         st.write(f"""
#         {best} achieved the highest R² score of {best_score:.2f},
#         meaning it explains the dataset variance better than other models.
#         """)

#     # =========================
#     # SHAP GLOBAL
#     # =========================
#     st.subheader("🔍 Global Feature Importance (SHAP)")

#     best_model = models[best]

#     try:
#         explainer = shap.TreeExplainer(best_model)

#         if task == "classification":
#             shap_vals = explainer.shap_values(X_test[:100])[1]
#         else:
#             shap_vals = explainer.shap_values(X_test[:100])    

#         importance = np.abs(shap_vals).mean(axis=0)
#         importance = np.asarray(importance, dtype=np.float64)

#         shap_df = pd.DataFrame({
#             "Feature": X.columns,
#             "Importance": importance
#         }).sort_values(by="Importance", ascending=False)

#         shap_df["Importance"] = shap_df["Importance"].astype(float)
#         st.bar_chart(shap_df.set_index("Feature"))

#         # =========================
#         # SHAP INTERPRETATION
#         # =========================
#         top_feature = shap_df.iloc[0]["Feature"]

#         st.write(f"""
#         🔑 The most influential feature is **{top_feature}**, 
#         meaning it has the highest impact on predictions.
#         """)

#     except Exception as e:
#         st.warning(f"SHAP failed: {e}")

#     # =========================
#     # FINAL INSIGHTS
#     # =========================
#     st.subheader("🧾 Final Insights")

#     st.write(f"""
#     - Dataset successfully processed and analyzed  
#     - Multiple models trained and evaluated  
#     - Best model: **{best}**  
#     - Feature importance analyzed using SHAP  
#     - System provides both predictive performance and interpretability  
#     """)

import streamlit as st
import pandas as pd
import numpy as np
import shap
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, r2_score
)


def run_mode2():

    st.title("📊 Mode 2: Dataset Analysis & Explainability")

    file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if file is None:
        st.info("Please upload a dataset to begin")
        return

    df = pd.read_csv(file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Diagnostics")
    st.write("Shape:", df.shape)
    st.write("Missing Values per Column:")
    st.write(df.isnull().sum())
    st.write("Data Types:")
    st.write(df.dtypes)

    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    st.subheader("🎯 Target Selection")
    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(target, axis=1)
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)

    num_classes = y.nunique()
    if y.dtype == "object" or num_classes <= 10:
        task = "classification"
    else:
        task = "regression"

    if task == "classification":
        if num_classes == 2:
            st.info("Binary Classification detected")
        else:
            st.info(f"Multiclass Classification detected ({num_classes} classes)")
    st.success(f"Detected Task: {task.upper()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    st.subheader("⚙️ Preprocessing Pipeline")
    st.markdown("""
    - Missing values removed
    - Categorical features encoded using One-Hot Encoding
    - Features scaled using StandardScaler
    - Data split into training (80%) and testing (20%)
    """)

    if task == "classification":
        models = {
            "Logistic": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True),
            "Random Forest": RandomForestClassifier()
        }
    else:
        models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Random Forest": RandomForestRegressor()
        }

    st.subheader("🤖 Model Training")

    results = {}
    predictions = {}

    for name, model in models.items():

        st.write(f"Training {name}...")

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred

        if task == "classification":
            avg = "binary" if num_classes == 2 else "weighted"
            results[name] = {
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, average=avg, zero_division=0),
                "Recall": recall_score(y_test, pred, average=avg, zero_division=0),
                "F1 Score": f1_score(y_test, pred, average=avg, zero_division=0)
            }
        else:
            results[name] = {
                "MAE": mean_absolute_error(y_test, pred),
                "R2": r2_score(y_test, pred)
            }

    res_df = pd.DataFrame(results).T
    for col in res_df.columns:
        res_df[col] = pd.to_numeric(res_df[col])

    st.subheader("📈 Model Performance Comparison")
    st.dataframe(res_df.astype(float))

    # =========================
    # ✅ FIX: BEST MODEL BEFORE CONFUSION MATRIX
    # =========================
    if task == "classification":
        best = res_df["Accuracy"].idxmax()
        best_pred = predictions[best]
        best_score = res_df.loc[best, "Accuracy"]
    else:
        best = res_df["R2"].idxmax()
        best_score = res_df.loc[best, "R2"]

    st.success(f"🏆 Best Model: {best}")

    # =========================
    # METRIC + CONFUSION MATRIX
    # =========================
    st.subheader("📊 Metric Interpretation")

    if task == "classification":

        st.subheader("📉 Confusion Matrix")

        show_norm = st.checkbox("Show Normalized Confusion Matrix")
        labels = sorted(y.unique())

        # ✅ FIX: now best_pred exists
        cm = confusion_matrix(y_test, best_pred, labels=labels)

        if show_norm:
            with np.errstate(all='ignore'):
                cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                cm_display = np.nan_to_num(cm_display)
            fmt = ".2f"
            cmap = "Greens"
            title = "Normalized Confusion Matrix"
        else:
            cm_display = cm
            fmt = "d"
            cmap = "Blues"
            title = "Confusion Matrix (Counts)"

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        st.pyplot(fig, use_container_width=True)

        # ✅ FIX: removed duplicate if
        if num_classes == 2:
            st.write("""
            Binary Classification Metrics:
            - Accuracy: Overall correctness
            - Precision: Correct positive predictions
            - Recall: Ability to detect positives
            - F1 Score: Balance between precision and recall
            """)
        else:
            st.write("""
            Multiclass Classification Metrics:
            - Accuracy: Overall correctness
            - Precision/Recall/F1: Computed using weighted average
            - Weighted average accounts for class imbalance
            """)

    else:
        st.write("""
        - MAE: Average prediction error  
        - R²: Proportion of variance explained by model  
        """)

    st.subheader("📊 Target Distribution")
    st.write(y.value_counts())

    st.subheader("🧠 Model Selection Justification")

    if task == "classification":
        st.write(f"""
        {best} achieved the highest accuracy of {best_score:.2f},
        indicating superior classification performance and generalization.
        """)
    else:
        st.write(f"""
        {best} achieved the highest R² score of {best_score:.2f},
        meaning it explains the dataset variance better than other models.
        """)

    st.subheader("🔍 Global Feature Importance (SHAP)")

    best_model = models[best]

    try:
        explainer = shap.TreeExplainer(best_model)

        if task == "classification":
            shap_vals = explainer.shap_values(X_test[:100])[1]
        else:
            shap_vals = explainer.shap_values(X_test[:100])

        importance = np.abs(shap_vals).mean(axis=0)
        importance = np.asarray(importance, dtype=np.float64)

        shap_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        shap_df["Importance"] = shap_df["Importance"].astype(float)
        st.bar_chart(shap_df.set_index("Feature"))

        top_feature = shap_df.iloc[0]["Feature"]

        st.write(f"""
        🔑 The most influential feature is **{top_feature}**
        """)

    except Exception as e:
        st.warning(f"SHAP failed: {e}")

    st.subheader("🧾 Final Insights")

    st.write(f"""
    - Dataset successfully processed and analyzed  
    - Multiple models trained and evaluated  
    - Best model: **{best}**  
    - Feature importance analyzed using SHAP  
    - System provides both predictive performance and interpretability  
    """)