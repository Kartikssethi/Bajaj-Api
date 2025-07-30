from flask import Blueprint, jsonify

# Import Blueprints from main and test
from main import gemini_bp as main_gemini_bp
from test import gemini_bp as test_gemini_bp

routes = Blueprint('routes', __name__)

# Register the Blueprints from main and test
routes.register_blueprint(main_gemini_bp, url_prefix='/main')
routes.register_blueprint(test_gemini_bp, url_prefix='/test')

@routes.route('/test-route', methods=['GET'])
def test_route():
    return jsonify({'message': 'Test route is working!'}) 