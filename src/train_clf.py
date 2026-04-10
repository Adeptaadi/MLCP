import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.preprocess_clf import preprocess_data
from src.models_clf.logistic import train_logistic
from src.models_clf.svm import train_svm
from src.models_clf.rf import train_rf

def train_classification():

    X, y, scaler, features = preprocess_data("data/processed.cleveland.data")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "logistic": train_logistic,
        "svm": train_svm,
        "rf": train_rf
    }

    trained_models = {}

    print("\n===== CLASSIFICATION =====\n")

    for name, func in models.items():
        model, y_pred = func(X_train, y_train, X_test)

        print(f"{name.upper()}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1:", f1_score(y_test, y_pred))
        print("-" * 30)

        trained_models[name] = model

    for name, model in trained_models.items():
        joblib.dump(model, f"models/{name}_model.pkl")

    joblib.dump(scaler, "models/scaler_clf.pkl")
    joblib.dump(features.tolist(), "models/feature_names.pkl")

    print("\nClassification models saved!")