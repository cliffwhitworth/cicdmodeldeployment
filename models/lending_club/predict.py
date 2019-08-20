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

# data
TESTING_DATA_FILE = 'lending_club_selected_features_test.csv'
TRAINING_DATA_FILE = 'lending_club_selected_features_train.csv'
TARGET = 'target'

# variables
FEATURES = ['loan_amnt', 'term', 'installment', 'grade', 'emp_length', 'home_ownership', 
            'annual_inc', 'verification_status', 'purpose', 'title', 'addr_state', 'dti', 
            'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec', 
            'revol_bal', 'revol_util', 'total_acc', 'last_credit_pull_d', 'pub_rec_bankruptcies', 'fico_average']

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['pub_rec_bankruptcies']

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['emp_length', 'title', 'revol_util', 'last_credit_pull_d']

# variables to log transform
NUMERICALS_LOG_VARS = ['loan_amnt', 'installment', 'annual_inc', 'dti', 
                       'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 
                       'revol_bal', 'total_acc', 'fico_average']

# categorical variables to encode
CATEGORICAL_VARS = ['term', 'grade', 'home_ownership', 'verification_status', 
                    'purpose', 'addr_state', 'earliest_cr_line', 'emp_length', 
                    'title', 'revol_util', 'last_credit_pull_d']

NUMERICAL_NA_NOT_ALLOWED = [
    feature for feature in FEATURES
    if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS
    if feature not in CATEGORICAL_VARS_WITH_NA
]

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

    return validated_data

pipeline_file_name = f'{PIPELINE_SAVE_FILE}.pkl'
_club_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)

    prediction = _club_pipe.predict(validated_data[FEATURES])

    results = {'predictions': prediction}

    return results