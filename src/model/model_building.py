import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from src.exception import MyException
import yaml
from src.logger import logging
import sys


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    # except pd.errors.ParserError as e:
    #     logging.error('Failed to parse the CSV file: %s', e)
    #     raise
    # except Exception as e:
    #     logging.error('Unexpected error occurred while loading the data: %s', e)
    #     raise
    except Exception as e:
            raise MyException(e,sys)

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        clf.fit(X_train, y_train)
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise MyException(e,sys)

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise MyException(e,sys)
    
def main():
    try:

        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)
        
        save_model(clf, 'models/model.pkl')
    except Exception as e:
        logging.error('Model building failed: %s', e)
        raise MyException(e, sys)

if __name__ == '__main__':
    main()