from typing import Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import pickle

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc-taxi-experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(data: Tuple[DictVectorizer, LinearRegression], **kwargs):
    dv, lr = data

    with mlflow.start_run():
        with open("dict_vectorizer.bin", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("dict_vectorizer.bin")

        mlflow.sklearn.log_model(lr, "model")

    print("OK")