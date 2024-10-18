import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException

class DataTransformation:
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Separate features and target
            target_column = "target_column"  # Replace with actual target column
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]
            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]

            logging.info("Applying data transformations...")
            num_features = X_train.select_dtypes(include=["int64", "float64"]).columns
            cat_features = X_train.select_dtypes(include=["object"]).columns

            numeric_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", numeric_pipeline, num_features),
                ("cat", categorical_pipeline, cat_features)
            ])

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

        except Exception as e:
            raise CustomException(e, sys)
