import os
from datetime import timedelta

class Config:
    """Base configuration class."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'cybereye_phishing_detector_2024'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///phishing_detector.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    
    # File upload configuration
    ALLOWED_EXTENSIONS = {
        'text': {'txt', 'csv', 'json'},
        'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'},
        'audio': {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac'},
        'video': {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
    }
    
    # Model configuration
    PHISHING_TEXT_MODEL = "martin-ha/toxic-comment-model"
    BACKUP_MODEL = "unitary/toxic-bert"
    
    # Default admin credentials
    ADMIN_EMAIL = 'admin@phishingdetector.com'
    ADMIN_PASSWORD = 'AdminPass@2024'
    
    @staticmethod
    def init_app(app):
        """Initialize app configuration."""
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

