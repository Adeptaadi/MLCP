import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_regression(path):

    df = pd.read_csv(path)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns