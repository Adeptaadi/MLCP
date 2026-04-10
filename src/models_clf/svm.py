from sklearn.svm import SVC

def train_svm(X_train, y_train, X_test):
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred