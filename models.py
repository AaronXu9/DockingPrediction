from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score
import numpy as np

# Define a random forest model
class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def cv_fit(self, X, y, val_set=None, cv=5):
        # scores = cross_val_score(self.model, X, y, cv=cv)
        self.model.fit(X, y)
        return self.model


# class RandomForestModel:
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42)

#     def fit(self, X_train, y_train):
#         self.model.fit(X_train, y_train)

#     def predict(self, X_test):
#         return self.model.predict(X_test)

#     def cross_val_score(self, X, y, val_set=None, cv=5):
#         if val_set is not None:
#             X_val, y_val = val_set
#             kf = KFold(n_splits=cv)
#             scores = []

#             for train_index, test_index in kf.split(X):
#                 clone_model = clone(self.model)
#                 X_train_fold, X_test_fold = X[train_index], X[test_index]
#                 y_train_fold, y_test_fold = y[train_index], y[test_index]

#                 # Here, we could use the validation set in fitting or adjusting the model,
#                 # but this is not typical for cross-validation
#                 clone_model.fit(X_train_fold, y_train_fold)

#                 # Evaluate the model on the test fold
#                 predictions = clone_model.predict(X_test_fold)
#                 score = r2_score(y_test_fold, predictions)
#                 scores.append(score)

#             return np.array(scores)
#         else:
#             return cross_val_score(self.model, X, y, cv=cv)
