# [Same as backend branch, plus this at the end]

from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import logging

# Initialize Flask app and config
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///farmiq.db'
app.config['DEBUG'] = True
db = SQLAlchemy(app)

# Example config dictionary
config = {
    'default': {
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///farmiq.db',
        'DEBUG': True
    }
}

# Setup logger
logger = logging.getLogger('farmiq')
logging.basicConfig(level=logging.INFO)

# Dummy data loader function
def load_dummy_data():
    pass  # Implement your dummy data loading logic here

# [DEPLOY] Health check endpoint
@app.route('/health')
def health():
    try:
        # Basic checks: DB connection, ML model load
        db.session.execute(db.text('SELECT 1'))
        from models.ml_model import PestPredictor
        predictor = PestPredictor()  # Quick load test
        return jsonify({'status': 'healthy', 'version': '1.0', 'timestamp': datetime.utcnow().isoformat()}), 200
    except Exception as e:
        logger.error(f"[DEPLOY] Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    app.config.from_object(config['default'])
    with app.app_context():
        db.create_all()
        load_dummy_data()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=app.config['DEBUG'])