from sklearn.linear_model import Lasso

def train_lasso(X_train, y_train, X_test):
    model = Lasso()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred