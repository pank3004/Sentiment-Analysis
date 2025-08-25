import pandas as pd
import numpy as np

import mlflow
import dagshub

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

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
    'experiment_name': 'Bow vs TfIdf'
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
        return df
    except Exception as e: 
        print(f'Error loading data: {e}')
        raise

### feature enginerrring: 
VECTORIZER={
    'BoW': CountVectorizer(), 
    'Tf-IDF': TfidfVectorizer()
}


ALGORITHMS = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    #'SVC': SVC(), 
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}


# Train and evaluate: 
logging.info('training and evaluating...')
def train_and_evaluate(df): 
    with mlflow.start_run(run_name='All Experiments') as parent_run: 
        for algo_name, algorithm in ALGORITHMS.items(): 
            for vec_name, vectorizer in VECTORIZER.items(): 
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run: 
                    try: 
                        ## feature extraction
                        logging.info('feature extraction....')
                        X=vectorizer.fit_transform(df['review'])
                        y=df['sentiment']
                        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=CONFIG['test_size'], random_state=42)

                        # log preprocessing params: 
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })

                        # model trainng
                        logging.info(f'model training>>{algo_name} and {vec_name}')
                        model=algorithm
                        model.fit(X_train, y_train)
                        logging.info(f'model training done on : {algo_name} and {vec_name}')

                        # log model params: 
                        log_model_params(algo_name, model)

                        # evaluation: 
                        logging.info(f'model prediction and evaluation: {algo_name} and {vec_name}')
                        y_pred=model.predict(X_test)
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred),
                            "recall": recall_score(y_test, y_pred),
                            "f1_score": f1_score(y_test, y_pred)
                        }   
                        logging.info(f'model prediction and evaluation done on  : {algo_name} and {vec_name}')
                        logging.info(f'logging evaluations metrics of {algo_name} with {vec_name}')
                        mlflow.log_metrics(metrics)
                        logging.info(f'logging evaluations metrics DONE of {algo_name} with {vec_name}')

                        # log model: 
                        input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                        mlflow.sklearn.log_model(model, 'model', input_example=input_example)

                        # Print results for verification
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")

                    except Exception as e:
                        print(f"Error in training {algo_name} with {vec_name}: {e}")
                        mlflow.log_param("error", str(e))

def log_model_params(algo_name, model):
    """Logs hyperparameters of the trained model to MLflow."""
    params_to_log = {}
    if algo_name == 'LogisticRegression':
        params_to_log["C"] = model.C
    # elif algo_name=='SVC': 
    #     params_to_log['']=model.C
    elif algo_name == 'MultinomialNB':
        params_to_log["alpha"] = model.alpha
    elif algo_name == 'XGBoost':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
    elif algo_name == 'RandomForest':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoosting':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth

    mlflow.log_params(params_to_log)

# ========================== EXECUTION ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)