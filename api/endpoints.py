from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from models.ml_model import MLPredictor
from models.rules_engine import get_recommendations
from utils.validators import validate_input
from utils.cache import cache
from models.user_auth import UserAuth
import json
import logging
from config import app

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

# [BACKEND] API Key check decorator
def require_api_key(f):
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != app.config['API_KEY']:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@api_bp.route('/weather/<location>')
@require_api_key
@cache.cached(timeout=600)  # 10 min cache
def get_weather(location):
    try:
        with open('data/dummy_weather.json', 'r') as f:
            data = json.load(f)
        weather = data.get(location, {'error': 'Location not found'})
        logger.info(f"[BACKEND] Weather API called for {location}")
        return jsonify(weather)
    except Exception as e:
        logger.error(f"[BACKEND] Weather API error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/soils/<soil_type>')
@require_api_key
def get_soils(soil_type):
    try:
        with open('data/dummy_soils.json', 'r') as f:
            data = json.load(f)
        soils = data.get(soil_type, {'error': 'Soil type not found'})
        logger.info(f"[BACKEND] Soils API called for {soil_type}")
        return jsonify(soils)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/recommendations', methods=['POST'])
@login_required
def post_recommendations():
    data = request.get_json()
    if not validate_input(data, ['crop', 'location', 'soil_type']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    crop = data['crop']
    location = data['location']
    soil_type = data['soil_type']
    
    # Fetch data
    with open('data/dummy_weather.json', 'r') as f:
        weather = json.load(f)[location]
    with open('data/dummy_sensors.json', 'r') as f:
        sensors = json.load(f)
    with open('data/dummy_soils.json', 'r') as f:
        soils = json.load(f)[soil_type]
    
    recs = get_recommendations(crop, soil_type, weather, sensors, soils)
    predictor = MLPredictor()
    recs['pest_risk'] = predictor.predict_pest_risk(sensors['moisture'], weather['temp'], crop)
    recs['yield_prediction'] = predictor.predict_yield(weather['rainfall'], sensors['moisture'], crop)
    
    logger.info(f"[BACKEND] Recommendations generated for {crop}")
    return jsonify(recs)

@api_bp.route('/pest/<crop>')
@require_api_key
def get_pest(crop):
    # Mock sensors/weather for API
    moisture = 25  # Default
    temp = 30
    predictor = MLPredictor()
    risk = predictor.predict_pest_risk(moisture, temp, crop)
    return jsonify({'crop': crop, 'pest_risk': risk, 'recommendation': f'Monitor {crop} for pests.'})

@api_bp.route('/yield/<crop>')
@require_api_key
def get_yield(crop):
    rainfall = 100  # Mock
    moisture = 40
    predictor = MLPredictor()
    pred = predictor.predict_yield(rainfall, moisture, crop)
    return jsonify({'crop': crop, 'predicted_yield': pred, 'unit': 'kg/hectare'})

@api_bp.route('/sustainability', methods=['POST'])
@login_required
def post_sustainability():
    data = request.get_json()
    if not validate_input(data, ['sensors', 'weather', 'soils']):
        return jsonify({'error': 'Missing data'}), 400
    
    predictor = MLPredictor()
    sust = predictor.calculate_sustainability(data['sensors'], data['weather'], data['soils'])
    return jsonify(sust)

# [BACKEND] Auth API (for mobile/app integration)
@api_bp.route('/auth/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user = UserAuth.authenticate(username, password)
    if user:
        return jsonify({'token': 'mock-jwt-token', 'user_id': user.id}), 200
    return jsonify({'error': 'Invalid credentials'}), 401