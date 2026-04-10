import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(path):

    df = pd.read_csv(path, header=None)

    df.columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "target"
    ]

    # Missing values
    df.replace("?", np.nan, inplace=True)

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Fill missing
    df.fillna(df.mean(), inplace=True)

    # Binary target
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    # Encoding
    categorical_cols = ["cp", "restecg", "slope", "thal"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Split
    X = df.drop("target", axis=1)
    y = df["target"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Final dataset shape:", df.shape)

    return X_scaled, y, scaler, X.columns