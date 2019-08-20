import pathlib
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from lending_club.config import config as cnf

def load_dataset(*, file_name: str
                 ) -> pd.DataFrame:
    _data = pd.read_csv(f'{cnf.DATASET_DIR}/{file_name}')
    return _data

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = cnf.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()

    # check for numerical variables with NA not seen during training
    if input_data[cnf.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=cnf.NUMERICAL_NA_NOT_ALLOWED)

    # check for categorical variables with NA not seen during training
    if input_data[cnf.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=cnf.CATEGORICAL_NA_NOT_ALLOWED)

    return validated_data

pipeline_file_name = f'{cnf.PIPELINE_SAVE_FILE}.pkl'
_club_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)

    prediction = _club_pipe.predict(validated_data[cnf.FEATURES])

    results = {'predictions': prediction}

    return results