import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException  # Ensure CustomException is implemented correctly

def save_object(file_path, obj):
    """Save a Python object to a file using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):
    """Train models with hyperparameter tuning using GridSearchCV and evaluate them."""
    try:
        report = {}
        for model_name, model in models.items():
            param_grid = param_grids.get(model_name, {})  # Handle missing parameter grids gracefully
            
            # Perform GridSearchCV
            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            # Predict and evaluate on train and test sets
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            # Store test score in the report
            report[model_name] = test_score

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Load a Python object from a pickle file."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
