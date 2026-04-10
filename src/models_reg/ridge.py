from sklearn.linear_model import Ridge

def train_ridge(X_train, y_train, X_test):
    model = Ridge()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred