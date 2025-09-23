import os
from dotenv import load_dotenv  # [DEPLOY] For .env support

load_dotenv()  # Load .env file

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
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-hackathon'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///farmiq.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    API_KEY = os.environ.get('FARM_IQ_API_KEY') or 'hackathon-api-key-123'
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'simple')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    # [DEPLOY] Rate limiting (e.g., 100/hour per IP)
    RATELIMIT_STORAGE_URL = os.environ.get('RATELIMIT_STORAGE_URL', 'memory://')

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///farmiq.db'

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://') if os.environ.get('DATABASE_URL') else 'sqlite:///farmiq.db'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig if os.environ.get('FLASK_ENV') != 'production' else ProductionConfig
}