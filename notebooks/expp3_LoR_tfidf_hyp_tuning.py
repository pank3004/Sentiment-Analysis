import pandas as pd
import numpy as np

import mlflow
import dagshub

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
import logging

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

import scipy.sparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")

# configarations: 

CONFIG={
    'data_path':'notebooks\IMDB Dataset.csv', 
    'test_size':0.2, 
    'mlflow_tracking_url': 'https://dagshub.com/pank3004/Sentiment-Analysis.mlflow', 
    'dagshub_repo_owner': 'pank3004', 
    'dagshub_repo_name': 'Sentiment-Analysis',
    'experiment_name': 'LoR Hyperparameter tuning'
}

# setup mlflow and dagshub: 

logging.info('set mlflow with dagshub')
mlflow.set_tracking_uri(CONFIG['mlflow_tracking_url'])
dagshub.init(repo_owner=CONFIG['dagshub_repo_owner'],
             repo_name=CONFIG['dagshub_repo_name'],
             mlflow=True)
mlflow.set_experiment(CONFIG['experiment_name'])



                                                    # text preprocessing
logging.info('preprocessing data...')
# convert in lowercase
def lower_case(text): 
    words=text.split(' ')
    words=[word.lower() for word in words]
    return ' '.join(words)


def remove_stop_words(text): 
    stop_words=set(stopwords.words('english'))
    text=[word for word in text.split() if word not in stop_words]
    return " ".join(text)

#removing numbers

def remove_numbers(text): 
    return re.sub(r'\d+', '', text)
            #or
# def removing_numbers(text): 
#     text="".join([char for char in text if not char.isdigit()])
#     return text


# removing punctuation: 
def remove_puntuations(text):
    translator=str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# removing urls: 
def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

# lemmetization
def lemmatization(text): 
    text=text.split()
    lemmatizer=WordNetLemmatizer()
    text=[lemmatizer.lemmatize(word, pos='v') for word in text]
    return " ".join(text)


def normalize_text(df):
    """Normalize the text data."""
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(remove_numbers)
        df['review'] = df['review'].apply(remove_puntuations)
        df['review'] = df['review'].apply(remove_urls)
        df['review'] = df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise



                                            # load and preprocess data: 
def load_data(file_path): 
    try: 
        df=pd.read_csv(file_path)
        df=normalize_text(df)
        df['sentiment']=df['sentiment'].map({'positive':1, 'negative':0})

        vectorizer=TfidfVectorizer()
        X=vectorizer.fit_transform(df['review'])
        y=df['sentiment']
        return train_test_split(X, y, test_size=CONFIG['test_size'], random_state=42), vectorizer
    except Exception as e: 
        print(f'Error loading data: {e}')
        raise


def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""
    
    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    

    with mlflow.start_run(): 
        grid_search=GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # log all hyperparameterr run: 
        for params, mean_score, std_score in zip(grid_search.cv_results_['params'],
                                                 grid_search.cv_results_['mean_test_score'],
                                                 grid_search.cv_results_['std_test_score']): 
            with mlflow.start_run(run_name=f'LR with params: {params}', nested=True): 
                model=LogisticRegression(**params)
                model.fit(X_train, y_train)

                y_pred=model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }

                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"\nBest Params: {best_params} | Best F1 Score: {best_f1:.4f}")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_data("notebooks\IMDB Dataset.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)