import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

class DataIngestion:
    def __init__(self):
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        try:
            logging.info("Reading the dataset...")
            df = pd.read_csv("notebook\Data\stud.csv")  # Update with correct path
            logging.info("Successfully read the dataset.")

            logging.info("Splitting the data into train and test sets...")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            train_df.to_csv(self.train_data_path, index=False)
            test_df.to_csv(self.test_data_path, index=False)

            logging.info("Data ingestion completed successfully.")
            return self.train_data_path, self.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
