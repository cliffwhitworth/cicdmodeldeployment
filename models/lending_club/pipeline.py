import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from lending_club.processing import preprocessors as pp
from lending_club.processing import features

# pipeline name
PIPELINE_NAME = 'lending_club'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output'

# paths
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

def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f'{DATASET_DIR}/{file_name}')
    return _data

def save_pipeline(*, pipeline_to_persist) -> None:
    save_file_name = f'{PIPELINE_SAVE_FILE}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

club_pipe = Pipeline(
    [
        ('categorical_imputer',
            pp.CategoricalImputer(variables=CATEGORICAL_VARS_WITH_NA)),
        ('numerical_imputer',
            pp.NumericalImputer(variables=NUMERICAL_VARS_WITH_NA)),
        ('rare_label_encoder',
            pp.RareLabelCategoricalEncoder(
                tol=0.01,
                variables=CATEGORICAL_VARS)),
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=CATEGORICAL_VARS)),
        ('scaler', MinMaxScaler()),
#         ('Linear_model', LogisticRegression(solver='lbfgs', class_weight='balanced'))
#         ('Linear_model', LogisticRegression(penalty='l2', tol=0.01, solver='saga'))
#         ('Linear_model', LogisticRegression(solver='lbfgs'))
        ('model', RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0))
    ]
)

def run_training() -> None:

    # read training data
    data = load_dataset(file_name=TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.1,
        random_state=0)  # we are setting the seed here

    club_pipe.fit(X_train[FEATURES],
                            y_train)

    save_pipeline(pipeline_to_persist=club_pipe)


if __name__ == '__main__':
    run_training()