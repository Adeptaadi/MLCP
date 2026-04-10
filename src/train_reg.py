import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.preprocess_reg import preprocess_regression
from src.models_reg.linear import train_linear
from src.models_reg.ridge import train_ridge
from src.models_reg.lasso import train_lasso
from src.models_reg.rf_reg import train_rf_reg

def train_regression():

    X, y, scaler, features = preprocess_regression("data/housing.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "linear": train_linear,
        "ridge": train_ridge,
        "lasso": train_lasso,
        "rf_reg": train_rf_reg
    }

    trained_models = {}

    print("\n===== REGRESSION =====\n")

    for name, func in models.items():
        model, y_pred = func(X_train, y_train, X_test)

        print(f"{name.upper()}")
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("MSE:", mean_squared_error(y_test, y_pred))
        print("R2:", r2_score(y_test, y_pred))
        print("-" * 30)

        trained_models[name] = model

    for name, model in trained_models.items():
        joblib.dump(model, f"models/{name}_model.pkl")

    joblib.dump(scaler, "models/scaler_reg.pkl")
    joblib.dump(features.tolist(), "models/feature_names_reg.pkl")

    print("\nRegression models saved!")