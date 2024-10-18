import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from src.utils import save_object, evaluate_models
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def split_data(self, train_array, test_array):
        try:
            logging.info("Splitting data into features and target...")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = self.split_data(train_array, test_array)

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
            }

            params = {
                "Random Forest": {"n_estimators": [10, 50, 100]},
                "Gradient Boosting": {"learning_rate": [0.01, 0.1]},
                "XGB Regressor": {"n_estimators": [10, 50], "learning_rate": [0.01, 0.1]},
                "CatBoost Regressor": {"iterations": [50, 100]},
            }

            logging.info("Evaluating models...")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with R2 score: {model_report[best_model_name]}")

            save_object(self.config.trained_model_path, best_model)

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException(e, sys)
