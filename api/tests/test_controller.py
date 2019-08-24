import json
import math

from regression_model.config import config as model_config
from regression_model.processing.data_management import load_dataset
from regression_model import __version__ as regression_model_version
from lending_club import __version__ as lending_club_version

from api import config as api_config
from api import __version__ as api_version

def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):
    # When
    response = flask_test_client.get('/version')

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['regression_model_version'] == regression_model_version
    assert response_json['lending_club_version'] == regression_model_version
    assert response_json['api_version'] == api_version


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # Given
    # Load the test data from the regression_model package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    test_data = load_dataset(file_name=model_config.TESTING_DATA_FILE)
    post_json = test_data[0:api_config.RUN_TESTS].to_json(orient='records')

    # When
    response = flask_test_client.post('/v1/predict/regression', json=post_json)

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    print(response_json.get('errors'))
    prediction = response_json['predictions'][0]
    response_version = response_json['version']
    assert math.ceil(prediction) == 112476
    assert response_version == regression_model_version
