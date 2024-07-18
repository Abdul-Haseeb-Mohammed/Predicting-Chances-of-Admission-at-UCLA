from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def train_MLP(x_train_scaled, y_train):
    MLP = MLPClassifier(hidden_layer_sizes=(2), batch_size=20, max_iter=100, random_state=123)
    MLP.fit(x_train_scaled,y_train)
    return MLP

def gridSearchCV(model, x_scaled, y,CV):
    # we will try different values for hyperparemeters
    params = {'batch_size':[20, 30, 40, 50],
            'hidden_layer_sizes':[(2,),(3,),(3,2)],
            'max_iter':[50, 70, 100]}
    # create a grid search
    grid = GridSearchCV(model, params, cv=CV, scoring='accuracy')
    grid.fit(x_scaled, y)
    return grid
