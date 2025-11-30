from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model, UserMixin):
    """User model for authentication."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    detections = db.relationship('Detection', backref='user', lazy=True, cascade='all, delete-orphan')
    queries = db.relationship('Query', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Set password hash."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password hash."""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user to dictionary."""
        return {
            'id': self.id,
            'email': self.email,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }
    
    def __repr__(self):
        return f'<User {self.email}>'

class Detection(db.Model):
    """Detection results model."""
    __tablename__ = 'detections'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    detection_type = db.Column(db.String(20), nullable=False)  # 'text', 'image', 'audio', 'video', 'url'
    input_data = db.Column(db.Text, nullable=True)  # Original input text or file info
    filename = db.Column(db.String(255), nullable=True)  # For file uploads
    file_path = db.Column(db.String(500), nullable=True)  # Path to uploaded file
    result_label = db.Column(db.String(50), nullable=False)  # 'Safe', 'Suspicious', 'Phishing', 'Malicious'
    confidence_score = db.Column(db.Float, nullable=False)  # 0-100 percentage
    safety_percentage = db.Column(db.Float, nullable=False)  # 0-100 percentage
    features_detected = db.Column(db.JSON, nullable=True)  # JSON of detected features
    analysis_details = db.Column(db.JSON, nullable=True)  # Detailed analysis results
    processing_time = db.Column(db.Float, nullable=True)  # Processing time in seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert detection to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'detection_type': self.detection_type,
            'input_data': self.input_data,
            'filename': self.filename,
            'result_label': self.result_label,
            'confidence_score': self.confidence_score,
            'safety_percentage': self.safety_percentage,
            'features_detected': self.features_detected,
            'analysis_details': self.analysis_details,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<Detection {self.id}: {self.result_label}>'

class Query(db.Model):
    """User queries/feedback model."""
    __tablename__ = 'queries'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(200), nullable=True)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='open')  # 'open', 'in_progress', 'resolved', 'closed'
    priority = db.Column(db.String(10), default='medium')  # 'low', 'medium', 'high', 'urgent'
    admin_response = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert query to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'email': self.email,
            'subject': self.subject,
            'message': self.message,
            'status': self.status,
            'priority': self.priority,
            'admin_response': self.admin_response,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<Query {self.id}: {self.subject}>'

class SystemLog(db.Model):
    """System activity logs."""
    __tablename__ = 'system_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.JSON, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert log to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'action': self.action,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<SystemLog {self.id}: {self.action}>'
