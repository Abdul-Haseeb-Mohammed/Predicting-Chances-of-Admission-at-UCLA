from sklearn.metrics import accuracy_score, confusion_matrix


def run_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    train_mae = accuracy_score(y_train, train_pred)
    train_confusion_matrix = confusion_matrix(y_train, train_pred)

    test_pred = model.predict(X_test)
    test_mae = accuracy_score(y_test, test_pred)
    test_confusion_matrix = confusion_matrix(y_test, test_pred)

    return train_mae, test_mae, train_confusion_matrix, test_confusion_matrix

