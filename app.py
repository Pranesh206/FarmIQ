from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from api.auth import UserAuth
from api.endpoints import api_bp, post_recommendations
from models.database import db, Farmer, RecommendationLog
# from models.ml_model import MLPredictor  # Uncomment if module exists
# from models.rules_engine import get_recommendations  # Uncomment if module exists
# import cache  # Uncomment if module exists
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
    # @cache.cached(timeout=300)  # 5 min cache
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
    recs = post_recommendations(farmer.crop, farmer.soil_type, weather, sensors, soils)
    
    # [BACKEND] ML Predictions
    predictor = predictor()
    pest_risk = predictor.predict_pest_risk(sensors['moisture'], weather['temp'], farmer.crop)
    yield_pred = predictor.predict_yield(weather['rainfall'], sensors['moisture'], farmer.crop)
    
    recs['pest_risk'] = pest_risk
    recs['yield_prediction'] = yield_pred
    recs['sustainability'] = predictor.calculate_sustainability(sensors, weather, soils)
    
    # [BACKEND] Log recommendation
    log = RecommendationLog(farmer_id=farmer.id, recommendations=json.dumps(recs), timestamp=datetime.utcnow())
    db.session.add(log)
    db.session.commit()
    
    return render_template('dashboard.html', farmer=farmer, recs=recs, weather=weather)

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

if __name__ == '__main__':
    with app.app_context():
        load_dummy_data()
    app.run(debug=app.config['DEBUG'])
from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
import json
import pandas as pd
import logging  # [DEV] For debug logging
# from models.ml_model import PestPredictor  # Uncomment if module exists
# from models.rules_engine import get_recommendations  # Uncomment if module exists
from config import Config
from datetime import datetime

# [DEV] Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

# Simple Farmer Model (same)
class Farmer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    crop = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    soil_type = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Load dummy data on startup (same)
@app.before_first_request
def load_dummy_data():
    with open('data/dummy_farmers.json', 'r') as f:
        farmers_data = json.load(f)
    for f in farmers_data:
        if not Farmer.query.filter_by(username=f['username']).first():
            farmer = Farmer(username=f['username'], crop=f['crop'], location=f['location'], soil_type=f['soil_type'])
            db.session.add(farmer)
    db.session.commit()
    logger.info("[DEV] Dummy data loaded successfully.")

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        username = request.form['username']
        crop = request.form['crop']
        location = request.form['location']
        soil_type = request.form['soil_type']
        farmer = Farmer(username=username, crop=crop, location=location, soil_type=soil_type)
        db.session.add(farmer)
        db.session.commit()
        session['user_id'] = farmer.id
        logger.info(f"[DEV] New farmer registered: {username}")
        return dashboard()
    except Exception as e:
        logger.error(f"[DEV] Registration error: {e}")
        return 'Registration failed', 500

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    farmer = Farmer.query.filter_by(username=username).first()
    if farmer:
        session['user_id'] = farmer.id
        logger.info(f"[DEV] Farmer logged in: {username}")
        return dashboard()
    logger.warning(f"[DEV] Login failed for: {username}")
    return 'User  not found', 404

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return render_template('login.html')
    farmer = Farmer.query.get(session['user_id'])
    
    # Mock data fetch (same, with logging)
    with open('data/dummy_weather.json', 'r') as f:
        weather = json.load(f)[farmer.location]  # Mock API
    logger.debug(f"[DEV] Weather data for {farmer.location}: {weather}")
    
    with open('data/dummy_sensors.json', 'r') as f:
        sensors = json.load(f)  # Mock IoT
    with open('data/dummy_soils.json', 'r') as f:
        soils = json.load(f)[farmer.soil_type]
    
    # Get recommendations (enhanced in rules_engine)
    # recs = get_recommendations(farmer.crop, farmer.soil_type, weather, sensors, soils)  # Uncomment if available
    
    # ML Pest prediction (with logging)
    # predictor = PestPredictor()
    # pest_risk = predictor.predict_pest_risk(sensors['moisture'], weather['temp'], farmer.crop)
    # logger.debug(f"[DEV] Pest risk prediction: {pest_risk}")
    
    # recs['pest_risk'] = pest_risk
    # recs['sustainability'] = {'water_saved': 20, 'carbon_footprint': 15, 'tip': 'Use organic fertilizers to reduce carbon by 10%.'}
    recs = {'sustainability': {'water_saved': 20, 'carbon_footprint': 15, 'tip': 'Use organic fertilizers to reduce carbon by 10%.'}}
    
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

@app.route('/api/weather/<location>')
def api_weather(location):
    with open('data/dummy_weather.json', 'r') as f:
        data = json.load(f)[location]
    logger.debug(f"[DEV] API call for weather: {location}")
    return jsonify(data)

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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        load_dummy_data()
    app.run(debug=app.config['DEBUG'])  # [DEV] Use config debug