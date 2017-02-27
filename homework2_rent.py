#!/usr/bin/env python
"""COMS 4995: Applied Machine Learning - Homework 2

Implements a linear model to predict the monthly rent of an apartment, assuming
that the market rate does not increase; i.e, the rent for a new tenant will be
the same as that for a current tenant.

The model is validated using 10 iterations of shuffle-split cross-validation on
the training set.

"""

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing_rent import X_train, X_test, y_train, y_test, X_test_raw

# Define Parameters
random_state = 0
test_size = 0.25
cv = ShuffleSplit(n_splits=10, test_size=test_size, random_state=random_state)

# Define Pipeline
pipe = make_pipeline(StandardScaler(), Ridge())

# Define Parameter Grid
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Implement the Model
model = GridSearchCV(pipe, param_grid=param_grid, cv=cv)

# Train the Model
model.fit(X_train, y_train)

# Evaluate the Model
def score_rent(model):
    """Returns the R^2 for the model on the test set"""
    # Computes the R^2 for the Test Set
    score = model.score(X_test, y_test)
    
    # Reports the Training and Test R^2 Respectively
    print("Training Set Score: {:.1f}%".format(model.best_score_ * 100))
    print("    Test Set Score: {:.1f}%".format(score * 100))
    
    return score

# Make Predictions
def predict_rent(X_test, y_test):
    """Returns the raw test data, the true labels, and the predicted labels"""
    # Make Predictions on the Test Set Using the Model
    y_pred = model.predict(X_test)
    
    return X_test_raw, y_test, y_pred
    
def main():
    score_rent(model)
    X_test_raw, y_labs, y_pred = predict_rent(X_test, y_test)
    
if __name__ == "__main__":
    main()