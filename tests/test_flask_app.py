import unittest
import os
import mlflow
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup MLflow credentials (if available)
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = "pank3004"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

            dagshub_url = "https://dagshub.com"
            repo_owner = "pank3004"
            repo_name = "Sentiment-Analysis"

            mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data=dict(text="I love this!"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Positive' in response.data or b'Negative' in response.data,
            "Response should contain either 'Positive' or 'Negative'"
        )


if __name__ == "__main__":
    unittest.main()
