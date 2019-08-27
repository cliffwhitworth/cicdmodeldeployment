import pathlib
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from lending_club.processing.data_management import load_pipeline
from lending_club.config import config
from lending_club.processing.validation import validate_inputs
from lending_club import __version__ as _version

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}.pkl'
_club_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)

    predictions = _club_pipe.predict(validated_data[config.FEATURES])

    results = {'predictions': predictions, 'version': _version}

    return results