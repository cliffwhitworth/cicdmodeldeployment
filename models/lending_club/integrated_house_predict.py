import pathlib
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

PIPELINE_NAME = 'lending_club'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output'

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_model'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# variables
FEATURES = ['MSSubClass', 'MSZoning', 'Neighborhood',
            'OverallQual', 'OverallCond', 'YearRemodAdd',
            'RoofStyle', 'MasVnrType', 'BsmtQual', 'BsmtExposure',
            'HeatingQC', 'CentralAir', '1stFlrSF', 'GrLivArea',
            'BsmtFullBath', 'KitchenQual', 'Fireplaces', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageCars', 'PavedDrive',
            'LotFrontage',
            # this one is only to calculate temporal variable:
            'YrSold']

# categorical variables to encode
CATEGORICAL_VARS = ['MSZoning', 'Neighborhood', 'RoofStyle', 'MasVnrType',
                    'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
                    'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'PavedDrive']

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['MasVnrType', 'BsmtQual', 'BsmtExposure',
                            'FireplaceQu', 'GarageType', 'GarageFinish']

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['LotFrontage']

NUMERICAL_NA_NOT_ALLOWED = [
    feature for feature in FEATURES
    if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS
    if feature not in CATEGORICAL_VARS_WITH_NA
]

# variables to log transform
NUMERICALS_LOG_VARS = ['LotFrontage', '1stFlrSF', 'GrLivArea']

def load_dataset(*, file_name: str
                 ) -> pd.DataFrame:
    _data = pd.read_csv(f'{DATASET_DIR}/{file_name}')
    return _data

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()

    # check for numerical variables with NA not seen during training
    if input_data[NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=NUMERICAL_NA_NOT_ALLOWED)

    # check for categorical variables with NA not seen during training
    if input_data[CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=CATEGORICAL_NA_NOT_ALLOWED)

    # check for values <= 0 for the log transformed variables
    if (input_data[NUMERICALS_LOG_VARS] <= 0).any().any():
        vars_with_neg_values = NUMERICALS_LOG_VARS[
            (input_data[NUMERICALS_LOG_VARS] <= 0).any()]
        validated_data = validated_data[
            validated_data[vars_with_neg_values] > 0]

    return validated_data

pipeline_file_name = f'{PIPELINE_SAVE_FILE}.pkl'
_price_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)

    prediction = _price_pipe.predict(validated_data[FEATURES])
    output = np.exp(prediction)

    results = {'predictions': output}

    return results