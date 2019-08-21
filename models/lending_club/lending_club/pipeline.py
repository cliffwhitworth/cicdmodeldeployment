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
from lending_club.config import config as cnf
from lending_club import __version__ as _version

def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f'{cnf.DATASET_DIR}/{file_name}')
    return _data

def save_pipeline(*, pipeline_to_persist) -> None:
    save_file_name = f'{cnf.PIPELINE_SAVE_FILE}.pkl'
    save_path = cnf.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

club_pipe = Pipeline(
    [
        ('categorical_imputer',
            pp.CategoricalImputer(variables=cnf.CATEGORICAL_VARS_WITH_NA)),
        ('numerical_imputer',
            pp.NumericalImputer(variables=cnf.NUMERICAL_VARS_WITH_NA)),
        ('rare_label_encoder',
            pp.RareLabelCategoricalEncoder(
                tol=0.01,
                variables=cnf.CATEGORICAL_VARS)),
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=cnf.CATEGORICAL_VARS)),
        ('scaler', MinMaxScaler()),
#         ('Linear_model', LogisticRegression(solver='lbfgs', class_weight='balanced'))
#         ('Linear_model', LogisticRegression(penalty='l2', tol=0.01, solver='saga'))
#         ('Linear_model', LogisticRegression(solver='lbfgs'))
        ('model', RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0))
    ]
)

def run_training() -> None:

    # read training data
    data = load_dataset(file_name=cnf.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[cnf.FEATURES],
        data[cnf.TARGET],
        test_size=0.1,
        random_state=0)  # we are setting the seed here

    club_pipe.fit(X_train[cnf.FEATURES],
                            y_train)

    save_pipeline(pipeline_to_persist=club_pipe)


if __name__ == '__main__':
    run_training()