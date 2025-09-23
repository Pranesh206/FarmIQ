import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-hackathon-backend'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///farmiq.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # [BACKEND] API Key for external calls (set in env for prod)
    API_KEY = os.environ.get('FARM_IQ_API_KEY') or 'hackathon-api-key-123'
    # [BACKEND] Caching config
    CACHE_TYPE = 'simple'  # In-memory; can be 'redis' for prod
    # [BACKEND] Debug mode
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')  # e.g., PostgreSQL

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}