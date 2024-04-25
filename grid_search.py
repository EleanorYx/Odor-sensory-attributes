import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid_rf = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20],  
    'min_samples_split': [2, 4, 6],  
    'min_samples_leaf': [1, 2, 3]  
}

param_grid_gbdt = {
    'n_estimators': [100, 200, 300],  
    'learning_rate': [0.01, 0.1, 0.2],  
    'max_depth': [3, 5, 7],  
}

param_grid_mlp = {
    'hidden_layer_sizes': [(50, 50), (100, 100)],  
    'activation': ['relu', 'tanh'],  
    'solver': ['adam', 'sgd'],  
    'max_iter': [200, 300]  
}

pipe_rf = Pipeline([
    ('scaler', StandardScaler()),  
    ('clf', Random Forest Classifier())
])

pipe_gbdt = Pipeline([
    ('scaler', StandardScaler()),  
    ('clf', Gradient Boosting Decision Trees())
])

pipe_mlp = Pipeline([
    ('scaler', StandardScaler()),  
    ('clf', MLPClassifier())
])

grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_gbdt = GridSearchCV(pipe_gbdt, param_grid_gbdt, cv=5, scoring='accuracy', n_jobs=-1)
grid_mlp = GridSearchCV(pipe_mlp, param_grid_mlp, cv=5, scoring='accuracy', n_jobs=-1)

grid_rf.fit(X_train, y_train)
grid_gbdt.fit(X_train, y_train)
grid_mlp.fit(X_train, y_train)


best_rf = grid_rf.best_params_
best_gbdt = grid_gbdt.best_params_
best_mlp = grid_mlp.best_params_

print("Best Parameters for Random Forest:", best_rf)
print("Best Parameters for Gradient Boosting:", best_gbdt)
print("Best Parameters for MLP Classifier:", best_mlp)

y_pred_rf = grid_rf.predict(X_test)
y_pred_gbdt = grid_gbdt.predict(X_test)
y_pred_mlp = grid_mlp.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_gbdt = accuracy_score(y_test, y_pred_gbdt)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print("Random Forest Test Accuracy:", accuracy_rf)
print("Gradient Boosting Test Accuracy:", accuracy_gbdt)
print("MLP Classifier Test Accuracy:", accuracy_mlp)
