#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import pandas as pd

import typer
import numpy as np
from typing_extensions import Annotated


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def main(
    year: Annotated[
        int, typer.Option(
            help="The year of Yellow Taxi Trip Records dataset (in YYYY format)."
        )
    ],
    month: Annotated[
        int, typer.Option(
            min=1,
            max=12,
            help="The month of Yellow Taxi Trip Records dataset (between 1 and 12)."
        )
    ],
    save_prediction: Annotated[
        bool, typer.Option(
            help="Save the predictions of the model to a file."
        )
    ] = False,
):

    df = read_data(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"Mean predicted duration: {np.mean(y_pred)}")

    if save_prediction:
        df_result = pd.DataFrame()
        ride_id = f"{year:04d}/{month:02d}_" + df.index.astype("str")
        df_result["ride_id"] = ride_id
        df_result["duration"] = y_pred

        output_file_name = f"prediction_yellow_tripdata_{year:04d}_{month:02d}.parquet"

        df_result.to_parquet(
            output_file_name,
            engine="pyarrow",
            compression=None,
            index=False,
        )


if __name__ == "__main__":
    typer.run(main)