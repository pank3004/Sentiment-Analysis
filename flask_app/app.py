from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
from src.exception import MyException
import logging
import sys

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text): 
    return re.sub(r'\d+', '', text)

# def removing_numbers(text):
#     """Remove numbers from the text."""
#     text = ''.join([char for char in text if not char.isdigit()])
#     return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    translator=str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# def removing_punctuations(text):
#     """Remove punctuations from the text."""
#     text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
#     text = text.replace('؛', "")
#     text = re.sub('\s+', ' ', text).strip()
#     return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    try:
        text = lower_case(text)
        text = remove_stop_words(text)
        text = removing_numbers(text)
        text = removing_punctuations(text)
        text = removing_urls(text)
        text = lemmatization(text)
        return text
    except Exception as e:
        logging.error("Error normalizing text", exc_info=True)
        raise MyException(e, sys)

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/pank3004/Sentiment-Analysis.mlflow')
# dagshub.init(repo_owner='pank3004', repo_name='Sentiment-Analysis', mlflow=True)
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "pank3004"
repo_name = "Sentiment-Analysis"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Initialize Flask app
app = Flask(__name__)


# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup

# ------------------- Model Loading -------------------
model_name = "my_model"

def get_latest_model_version(model_name):
    try:
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Staging"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["None"])
        return latest_version[0].version if latest_version else None
    except Exception as e:
        logging.error("Error fetching model version", exc_info=True)
        raise MyException(e, sys)


try:
    model_version = get_latest_model_version(model_name)
    model_uri = f'models:/{model_name}/{model_version}'
    logging.info(f"Fetching model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error("Error loading model/vectorizer", exc_info=True)
    raise MyException(e, sys)

# ------------------- Routes -------------------
@app.route("/")
def home():
    try:
        REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
        start_time = time.time()
        response = render_template("index.html", result=None)
        REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
        logging.info("Home page rendered.")
        return response
    except Exception as e:
        logging.error("Error in home route", exc_info=True)
        raise MyException(e, sys)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
        start_time = time.time()

        text = request.form["text"]
        logging.info(f"Received text: {text}")

        # Clean text
        text = normalize_text(text)
        logging.info(f"Normalized text: {text}")

        # Convert to features
        features = vectorizer.transform([text])
        features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

        # Predict
        result = model.predict(features_df)
        prediction = result[0]
        logging.info(f"Prediction made: {prediction}")

        # Increment prediction count metric
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

        # Measure latency
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return render_template("index.html", result=prediction)

    except Exception as e:
        logging.error("Error during prediction", exc_info=True)
        raise MyException(e, sys)


@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}
    except Exception as e:
        logging.error("Error generating metrics", exc_info=True)
        raise MyException(e, sys)


if __name__ == "__main__":
    try:
        logging.info("Starting Flask server...")
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as e:
        logging.error("Error starting Flask server", exc_info=True)
        raise MyException(e, sys)
    
