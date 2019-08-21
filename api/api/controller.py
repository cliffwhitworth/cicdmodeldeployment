from flask import Blueprint, request

from regression_model import __version__ as regression_model_version
from lending_club import __version__ as lending_club_version
from api import __version__ 



prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        return 'ok'

@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'regression_model_version': regression_model_version,
			'lending_club_version': lending_club_version,
                        'api_version': api_version})
