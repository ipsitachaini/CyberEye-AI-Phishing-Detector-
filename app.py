import os
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash

# Import our custom modules
from config import config
from models import db, User, Detection, Query, SystemLog
from utils import (
    PhishingDetector, allowed_file, secure_filename_custom, 
    get_file_type, get_file_hash, analyze_text, analyze_file_content
)

def create_app(config_name='default'):
    """Application factory pattern."""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    
    # Initialize Login Manager
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Define create_default_admin function BEFORE using it
    def create_default_admin():
        """Create default admin user if it doesn't exist."""
        admin_email = app.config['ADMIN_EMAIL']
        admin_password = app.config['ADMIN_PASSWORD']
        
        existing_admin = User.query.filter_by(email=admin_email).first()
        if not existing_admin:
            admin_user = User(
                email=admin_email,
                is_admin=True
            )
            admin_user.set_password(admin_password)
            db.session.add(admin_user)
            db.session.commit()
            print(f"Created default admin user: {admin_email}")
    
    # Create database tables and default admin
    with app.app_context():
        db.create_all()
        create_default_admin()
    
    def log_activity(action, details=None):
        """Log user activity."""
        try:
            log = SystemLog(
                user_id=current_user.id if current_user.is_authenticated else None,
                action=action,
                details=details,
                ip_address=request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR')),
                user_agent=request.environ.get('HTTP_USER_AGENT')
            )
            db.session.add(log)
            db.session.commit()
        except Exception as e:
            print(f"Error logging activity: {e}")
    
    # Routes
    @app.route('/')
    def index():
        """Home page - redirect based on authentication status."""
        if current_user.is_authenticated:
            if current_user.is_admin:
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        return redirect(url_for('login'))
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """User login page."""
        if current_user.is_authenticated:
            return redirect(url_for('index'))
        
        if request.method == 'POST':
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            remember = bool(request.form.get('remember'))
            
            if not email or not password:
                flash('Please provide both email and password.', 'warning')
                return render_template('login.html')
            
            user = User.query.filter_by(email=email).first()
            
            if user and user.check_password(password) and user.is_active:
                login_user(user, remember=remember)
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                log_activity('User login', {'email': email})
                
                next_page = request.args.get('next')
                if next_page:
                    return redirect(next_page)
                return redirect(url_for('index'))
            else:
                flash('Invalid email or password.', 'danger')
                log_activity('Failed login attempt', {'email': email})
        
        return render_template('login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """User registration page."""
        if current_user.is_authenticated:
            return redirect(url_for('index'))
        
        if request.method == 'POST':
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')
            
            # Validation
            if not email or not password or not confirm_password:
                flash('All fields are required.', 'warning')
                return render_template('register.html')
            
            if password != confirm_password:
                flash('Passwords do not match.', 'warning')
                return render_template('register.html')
            
            if len(password) < 6:
                flash('Password must be at least 6 characters long.', 'warning')
                return render_template('register.html')
            
            # Check if user already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already registered. Please login.', 'info')
                return redirect(url_for('login'))
            
            # Create new user
            new_user = User(email=email)
            new_user.set_password(password)
            
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! Please login.', 'success')
            log_activity('User registration', {'email': email})
            
            return redirect(url_for('login'))
        
        return render_template('register.html')
    
    @app.route('/logout')
    @login_required
    def logout():
        """User logout."""
        log_activity('User logout')
        logout_user()
        flash('You have been logged out.', 'info')
        return redirect(url_for('login'))
    
    @app.route('/user/dashboard')
    @login_required
    def user_dashboard():
        """User dashboard with recent detections and statistics."""
        # Get recent detections
        recent_detections = Detection.query.filter_by(user_id=current_user.id)\
                                         .order_by(Detection.created_at.desc())\
                                         .limit(10).all()
        
        # Get statistics
        total_detections = Detection.query.filter_by(user_id=current_user.id).count()
        
        # Count by type
        text_count = Detection.query.filter_by(user_id=current_user.id, detection_type='text').count()
        image_count = Detection.query.filter_by(user_id=current_user.id, detection_type='image').count()
        audio_count = Detection.query.filter_by(user_id=current_user.id, detection_type='audio').count()
        video_count = Detection.query.filter_by(user_id=current_user.id, detection_type='video').count()
        
        # Count by result
        safe_count = Detection.query.filter_by(user_id=current_user.id, result_label='Safe').count()
        suspicious_count = Detection.query.filter_by(user_id=current_user.id, result_label='Suspicious').count()
        phishing_count = Detection.query.filter_by(user_id=current_user.id, result_label='Phishing').count()
        
        # Get user queries
        user_queries = Query.query.filter_by(user_id=current_user.id)\
                                .order_by(Query.created_at.desc())\
                                .limit(5).all()
        
        stats = {
            'total_detections': total_detections,
            'text_count': text_count,
            'image_count': image_count,
            'audio_count': audio_count,
            'video_count': video_count,
            'safe_count': safe_count,
            'suspicious_count': suspicious_count,
            'phishing_count': phishing_count
        }
        
        return render_template('user_dashboard.html', 
                             detections=recent_detections,
                             stats=stats,
                             queries=user_queries)
    
    @app.route('/admin/dashboard')
    @login_required
    def admin_dashboard():
        """Admin dashboard with system statistics."""
        if not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('user_dashboard'))
        
        # Get system statistics
        total_users = User.query.filter_by(is_admin=False).count()
        total_detections = Detection.query.count()
        total_queries = Query.query.count()
        
        # Recent activity
        recent_users = User.query.filter_by(is_admin=False)\
                                .order_by(User.created_at.desc())\
                                .limit(10).all()
        
        recent_detections = Detection.query.order_by(Detection.created_at.desc())\
                                         .limit(15).all()
        
        recent_queries = Query.query.order_by(Query.created_at.desc())\
                                   .limit(10).all()
        
        # System logs
        recent_logs = SystemLog.query.order_by(SystemLog.created_at.desc())\
                                    .limit(20).all()
        
        # Detection statistics
        detection_stats = {
            'safe': Detection.query.filter_by(result_label='Safe').count(),
            'suspicious': Detection.query.filter_by(result_label='Suspicious').count(),
            'phishing': Detection.query.filter_by(result_label='Phishing').count(),
            'total': total_detections
        }
        
        return render_template('admin_dashboard.html',
                             total_users=total_users,
                             total_detections=total_detections,
                             total_queries=total_queries,
                             recent_users=recent_users,
                             recent_detections=recent_detections,
                             recent_queries=recent_queries,
                             recent_logs=recent_logs,
                             detection_stats=detection_stats)
    
    @app.route('/analyze/text', methods=['POST'])
    @login_required
    def analyze_text_route():
        """Analyze text for phishing content."""
        text_input = request.form.get('text_input', '').strip()
        
        if not text_input:
            flash('Please provide text to analyze.', 'warning')
            return redirect(url_for('user_dashboard'))
        
        if len(text_input) > 5000:
            flash('Text is too long. Maximum 5000 characters allowed.', 'warning')
            return redirect(url_for('user_dashboard'))
        
        try:
            # Initialize detector and perform analysis
            detector = PhishingDetector()
            result = detector.detect_text_phishing(text_input)
            
            # Save detection result
            detection = Detection(
                user_id=current_user.id,
                detection_type='text',
                input_data=text_input[:500],  # Store first 500 chars
                result_label=result['label'],
                confidence_score=result['confidence'],
                safety_percentage=result['safety_percentage'],
                features_detected=result.get('features', []),
                analysis_details={
                    'text_length': len(text_input),
                    'analysis_method': 'text_analysis'
                },
                processing_time=result.get('processing_time', 0)
            )
            
            db.session.add(detection)
            db.session.commit()
            
            log_activity('Text analysis', {
                'result': result['label'],
                'confidence': result['confidence']
            })
            
            flash(f'Analysis complete: {result["label"]} (Safety: {result["safety_percentage"]:.1f}%)', 
                  'success' if result['label'] == 'Safe' else 'warning')
            
        except Exception as e:
            flash(f'Error analyzing text: {str(e)}', 'danger')
            log_activity('Text analysis error', {'error': str(e)})
        
        return redirect(url_for('user_dashboard'))
    
    @app.route('/analyze/file', methods=['POST'])
    @login_required
    def analyze_file_route():
        """Analyze uploaded file for phishing content."""
        if 'file_input' not in request.files:
            flash('No file selected.', 'warning')
            return redirect(url_for('user_dashboard'))
        
        file = request.files['file_input']
        
        if file.filename == '':
            flash('No file selected.', 'warning')
            return redirect(url_for('user_dashboard'))
        
        if not allowed_file(file.filename):
            flash('File type not supported.', 'warning')
            return redirect(url_for('user_dashboard'))
        
        try:
            # Secure filename and save file
            filename = secure_filename_custom(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Determine file type and analyze
            file_type = get_file_type(file.filename)
            
            # Initialize detector and analyze file
            detector = PhishingDetector()
            result = detector.analyze_file(file_path, file_type)
            
            # Calculate file hash for deduplication
            file_hash = get_file_hash(file_path)
            
            # Save detection result
            detection = Detection(
                user_id=current_user.id,
                detection_type=file_type,
                filename=file.filename,
                file_path=file_path,
                result_label=result['label'],
                confidence_score=result['confidence'],
                safety_percentage=result['safety_percentage'],
                features_detected=result.get('features', []),
                analysis_details={
                    'file_type': file_type,
                    'file_hash': file_hash,
                    'original_filename': file.filename,
                    'file_size': os.path.getsize(file_path),
                    'extracted_text': result.get('extracted_text', ''),
                    'transcribed_text': result.get('transcribed_text', '')
                },
                processing_time=result.get('processing_time', 0)
            )
            
            db.session.add(detection)
            db.session.commit()
            
            log_activity('File analysis', {
                'filename': file.filename,
                'file_type': file_type,
                'result': result['label'],
                'confidence': result['confidence']
            })
            
            flash(f'File analysis complete: {result["label"]} (Safety: {result["safety_percentage"]:.1f}%)', 
                  'success' if result['label'] == 'Safe' else 'warning')
            
        except Exception as e:
            flash(f'Error analyzing file: {str(e)}', 'danger')
            log_activity('File analysis error', {'error': str(e)})
        
        return redirect(url_for('user_dashboard'))
    
    @app.route('/query', methods=['GET', 'POST'])
    @login_required
    def submit_query():
        """Submit query/feedback."""
        if request.method == 'POST':
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip()
            subject = request.form.get('subject', '').strip()
            message = request.form.get('message', '').strip()
            
            if not all([name, email, message]):
                flash('Name, email, and message are required.', 'warning')
                return render_template('query.html')
            
            query = Query(
                user_id=current_user.id,
                name=name,
                email=email,
                subject=subject,
                message=message
            )
            
            db.session.add(query)
            db.session.commit()
            
            log_activity('Query submitted', {'subject': subject})
            flash('Your query has been submitted successfully.', 'success')
            
            return redirect(url_for('user_dashboard'))
        
        return render_template('query.html')
    
    @app.route('/about')
    def about():
        """About page."""
        return render_template('about.html')
    
    @app.route('/api/analyze', methods=['POST'])
    @login_required
    def api_analyze():
        """API endpoint for analysis."""
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return jsonify({'error': 'No text provided'}), 400
            
            text = data['text'].strip()
            if not text:
                return jsonify({'error': 'Empty text provided'}), 400
            
            # Initialize detector and analyze
            detector = PhishingDetector()
            result = detector.detect_text_phishing(text)
            
            # Save to database
            detection = Detection(
                user_id=current_user.id,
                detection_type='text',
                input_data=text[:500],
                result_label=result['label'],
                confidence_score=result['confidence'],
                safety_percentage=result['safety_percentage'],
                features_detected=result.get('features', []),
                processing_time=result.get('processing_time', 0)
            )
            
            db.session.add(detection)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'result': result,
                'detection_id': detection.id
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/history')
    @login_required
    def api_history():
        """Get user's detection history."""
        try:
            detections = Detection.query.filter_by(user_id=current_user.id)\
                                      .order_by(Detection.created_at.desc())\
                                      .limit(50).all()
            
            return jsonify({
                'success': True,
                'detections': [d.to_dict() for d in detections]
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/static/<path:filename>')
    def static_files(filename):
        """Serve static files."""
        return send_from_directory('static', filename)
    
    @app.route('/uploads/<path:filename>')
    @login_required
    def uploaded_files(filename):
        """Serve uploaded files (admin only)."""
        if not current_user.is_admin:
            return "Access denied", 403
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('404.html') if os.path.exists('templates/404.html') else "Page not found", 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('500.html') if os.path.exists('templates/500.html') else "Internal server error", 500
    
    return app

if __name__ == '__main__':
    app = create_app('development')
    
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("🛡️  Starting Phishing Detection Application...")
    print("=" * 50)
    print("📧 Default admin credentials:")
    print(f"   Email: {app.config['ADMIN_EMAIL']}")
    print(f"   Password: {app.config['ADMIN_PASSWORD']}")
    print("=" * 50)
    print("🌐 Server starting at: http://localhost:5000")
    print("📁 Upload directory: uploads/")
    print("💾 Database: phishing_detector.db")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)