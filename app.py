from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from api.auth import UserAuth
from api.endpoints import api_bp
from models.database import db, Farmer, RecommendationLog
from models.ml_model import MLPredictor
from models.rules_engine import get_recommendations
from utils.cache import cache
import json
import logging
from config import config
from datetime import datetime

# [BACKEND] Setup logging
logging.basicConfig(level=logging.INFO if config['default'].DEBUG else logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(config['default'])
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Register API blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return UserAuth.get_user(user_id)

# Load dummy data on startup
@app.before_first_request
def load_dummy_data():
    with app.app_context():
        db.create_all()
        with open('data/dummy_farmers.json', 'r') as f:
            farmers_data = json.load(f)
        for f in farmers_data:
            if not Farmer.query.filter_by(username=f['username']).first():
                farmer = Farmer(**f)
                db.session.add(farmer)
        db.session.commit()
        logger.info("[BACKEND] Dummy data loaded.")

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']  # [BACKEND] Added password
        crop = request.form['crop']
        location = request.form['location']
        soil_type = request.form['soil_type']
        
        # [BACKEND] Auth integration
        user = UserAuth.create_user(username, password)
        if user:
            farmer = Farmer(username=username, crop=crop, location=location, soil_type=soil_type, user_id=user.id)
            db.session.add(farmer)
            db.session.commit()
            login_user(user)
            logger.info(f"[BACKEND] User registered: {username}")
            return redirect(url_for('dashboard'))
        return 'Registration failed', 400
    return render_template('register.html')  # [BACKEND] Separate register template

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = UserAuth.authenticate(username, password)
        if user:
            login_user(user)
            logger.info(f"[BACKEND] User logged in: {username}")
            return redirect(url_for('dashboard'))
        return 'Invalid credentials', 401
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    farmer = Farmer.query.filter_by(user_id=current_user.id).first()
    if not farmer:
        return redirect(url_for('register'))
    
    # [BACKEND] Cached data fetch
    @cache.cached(timeout=300)  # 5 min cache
    def get_cached_data(location):
        with open('data/dummy_weather.json', 'r') as f:
            weather = json.load(f)[location]
        with open('data/dummy_sensors.json', 'r') as f:
            sensors = json.load(f)
        with open('data/dummy_soils.json', 'r') as f:
            soils = json.load(f)[farmer.soil_type]
        return weather, sensors, soils
    
    weather, sensors, soils = get_cached_data(farmer.location)
    
    # Recommendations
    recs = get_recommendations(farmer.crop, farmer.soil_type, weather, sensors, soils)
    
    # [BACKEND] ML Predictions
    predictor = MLPredictor()
    pest_risk = predictor.predict_pest_risk(sensors['moisture'], weather['temp'], farmer.crop)
    yield_pred = predictor.predict_yield(weather['rainfall'], sensors['moisture'], farmer.crop)
    
    recs['pest_risk'] = pest_risk
    recs['yield_prediction'] = yield_pred
    recs['sustainability'] = predictor.calculate_sustainability(sensors, weather, soils)
    
    # [BACKEND] Log recommendation
    log = RecommendationLog(farmer_id=farmer.id, recommendations=json.dumps(recs), timestamp=datetime.utcnow())
    db.session.add(log)
    db.session.commit()
    
    # [DEV] Integration status (mock connected sources)
    status = {
        'weather': 'Connected (Mock API)',
        'sensors': 'Simulated (IoT)',
        'soils': 'Integrated (AgriStack-like)'
    }
    
    return render_template('dashboard.html', farmer=farmer, recs=recs, weather=weather, status=status)

@app.route('/advice/<category>')
def advice(category):
    # Detailed advice (same, with logging)
    advices = {
        'irrigation': 'Irrigate every 3 days based on soil moisture.',
        'fertilizer': 'Apply NPK 10-20-10 for rice.',
        'pest': 'Watch for aphids; use neem oil.',
        'sowing': 'Sow rice in June for monsoon.'
    }
    logger.debug(f"[DEV] Advice requested for: {category}")
    return render_template('advice.html', category=category, advice=advices.get(category, 'General tips.'))

# [BACKEND] Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"[BACKEND] Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# [BACKEND] API docs route (simple)
@app.route('/api/docs')
def api_docs():
    docs = """
    FarmIQ API Documentation:
    - GET /api/weather/<location>: Weather data
    - POST /api/recommendations: Personalized advice (JSON body: {'crop': 'rice', 'location': 'Punjab'})
    - GET /api/pest/<crop>: Pest risk prediction
    - GET /api/soils/<soil_type>: AgriStack-like soil data
    - POST /api/auth/login: User login (JSON: {'username': '', 'password': ''})
    API Key: Required in header 'X-API-Key'
    """
    return docs, 200, {'Content-Type': 'text/plain'}

# [NEW DEV] Debug route to dump raw data
@app.route('/debug/data')
def debug_data():
    try:
        with open('data/dummy_weather.json', 'r') as f:
            weather = json.load(f)
        with open('data/dummy_sensors.json', 'r') as f:
            sensors = json.load(f)
        with open('data/dummy_soils.json', 'r') as f:
            soils = json.load(f)
        debug_info = {'weather': weather, 'sensors': sensors, 'soils': soils}
        logger.info("[DEV] Debug data dump requested.")
        return jsonify(debug_info)
    except Exception as e:
        logger.error(f"[DEV] Debug error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/weather/<location>')
def api_weather(location):
    with open('data/dummy_weather.json', 'r') as f:
        data = json.load(f)[location]
    logger.debug(f"[DEV] API call for weather: {location}")
    return jsonify(data)

if __name__ == '__main__':
    with app.app_context():
        load_dummy_data()
    app.run(debug=app.config['DEBUG'])
