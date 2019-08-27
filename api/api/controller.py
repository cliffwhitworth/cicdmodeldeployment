import json

from flask import Blueprint, request, jsonify

from regression_model.predict import make_prediction as make_regression_prediction
from regression_model import __version__ as regression_version

from lending_club.predict import make_prediction as make_club_prediction
from lending_club import __version__ as club_version

from api.config import get_logger
from api.regression_validation import validate_inputs as validate_regression_inputs
from api.club_validation import validate_inputs as validate_club_inputs
from api import __version__ as api_version

_logger = get_logger(logger_name=__name__)


prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'regression_version': regression_version,
			'club_version': club_version,
                        'api_version': api_version})


@prediction_app.route('/v1/predict/regression', methods=['POST'])
def predict_regression():
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        _logger.debug(f'Inputs: {json_data}')

        # Step 2: Validate the input using marshmallow schema
        input_data, errors = validate_regression_inputs(input_data=json_data)

        # Step 3: Model prediction
        result = make_regression_prediction(input_data=input_data)
        _logger.debug(f'Outputs: {result}')

        # Step 4: Convert numpy ndarray to list
        predictions = result.get('predictions').tolist()
        version = result.get('version')

        # Step 5: Return the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})

@prediction_app.route('/v1/predict/club', methods=['POST'])
def predict_club():
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        _logger.debug(f'Inputs: {json_data}')

        # Step 2: Validate the input using marshmallow schema
        input_data, errors = validate_club_inputs(input_data=json_data)

        # Step 3: Model prediction
        result = make_club_prediction(input_data=input_data)
        _logger.debug(f'Outputs: {result}')

        # Step 4: Convert numpy ndarray to list
        predictions = result.get('predictions').tolist()
        version = result.get('version')

        # Step 5: Return the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})
