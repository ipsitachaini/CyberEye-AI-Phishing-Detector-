# import os
# import re
# import time
# import json
# import logging
# import hashlib
# from datetime import datetime
# from typing import Dict, List, Tuple, Optional, Any
# import numpy as np
# import pandas as pd

# # ML/AI Libraries
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# import joblib

# # Media processing libraries
# try:
#     import pytesseract
#     from PIL import Image
#     OCR_AVAILABLE = True
#     # Set Tesseract path for Windows (adjust if needed)
#     if os.name == 'nt':  # Windows
#         pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# except ImportError:
#     OCR_AVAILABLE = False
#     print("Warning: OCR libraries not available")

# try:
#     import whisper
#     WHISPER_AVAILABLE = True
# except ImportError:
#     WHISPER_AVAILABLE = False
#     print("Warning: Whisper not available")

# try:
#     import cv2
#     import librosa
#     import soundfile as sf
#     MEDIA_PROCESSING_AVAILABLE = True
# except ImportError:
#     MEDIA_PROCESSING_AVAILABLE = False
#     print("Warning: Media processing libraries not available")

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class PhishingDetector:
#     """Enhanced phishing detection class with improved accuracy."""
    
#     def __init__(self):
#         self.text_model = None
#         self.text_tokenizer = None
#         self.backup_classifier = None
#         self.whisper_model = None
#         self.tfidf_vectorizer = None
#         self.load_models()
    
#     def load_models(self):
#         """Load all detection models."""
#         try:
#             # Try to load a better phishing detection model
#             model_name = "martin-ha/toxic-comment-model"
#             self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.text_model = AutoModelForSequenceClassification.from_pretrained(model_name)
#             self.text_model.eval()
#             logger.info(f"Loaded text model: {model_name}")
#         except Exception as e:
#             logger.warning(f"Could not load main text model: {e}")
#             self.setup_backup_classifier()
        
#         # Load Whisper for audio processing
#         if WHISPER_AVAILABLE:
#             try:
#                 self.whisper_model = whisper.load_model("tiny")  # Use tiny model for faster loading
#                 logger.info("Loaded Whisper model")
#             except Exception as e:
#                 logger.warning(f"Could not load Whisper model: {e}")
    
#     def setup_backup_classifier(self):
#         """Enhanced backup classifier with better phishing detection."""
#         try:
#             # Enhanced training data with more diverse examples
#             safe_texts = [
#                 "Thank you for your recent purchase",
#                 "Meeting scheduled for tomorrow at 2 PM",
#                 "Project update and progress report attached",
#                 "Welcome to our monthly newsletter",
#                 "Your order #12345 has been confirmed and shipped",
#                 "Reminder: Team meeting this Friday",
#                 "Happy birthday! Hope you have a wonderful day",
#                 "Invoice for services rendered in October",
#                 "Please find the requested documents attached",
#                 "Weather forecast for this weekend",
#             ]
            
#             phishing_texts = [
#                 "URGENT: Your account will be suspended immediately",
#                 "Verify your PayPal account now or lose access",
#                 "Click here to claim your $1000 prize winner",
#                 "Your bank account needs immediate verification",
#                 "Security alert: Suspicious login detected. Act now!",
#                 "Congratulations! You've won the lottery. Click to claim",
#                 "Your Apple ID has been locked. Verify immediately",
#                 "Amazon: Your account has been compromised. Reset password",
#                 "IRS: You owe back taxes. Pay now to avoid arrest",
#                 "Microsoft: Your computer is infected. Call now",
#                 "Bitcoin opportunity: Make $5000 daily guaranteed",
#                 "Your credit card expires today. Update information",
#                 "Facebook: Your account will be deleted. Confirm identity",
#                 "Government refund pending. Provide bank details",
#                 "You have 24 hours to verify your identity or account will be closed"
#             ]
            
#             # Prepare training data
#             X_train = safe_texts + phishing_texts
#             y_train = [0] * len(safe_texts) + [1] * len(phishing_texts)
            
#             # Enhanced TF-IDF with better parameters
#             self.tfidf_vectorizer = TfidfVectorizer(
#                 max_features=2000,
#                 stop_words='english',
#                 ngram_range=(1, 2),  # Include bigrams
#                 min_df=1,
#                 lowercase=True
#             )
#             X_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
            
#             # Use logistic regression with balanced class weights
#             self.backup_classifier = LogisticRegression(
#                 class_weight='balanced',
#                 random_state=42,
#                 max_iter=1000
#             )
#             self.backup_classifier.fit(X_tfidf, y_train)
            
#             logger.info("Enhanced backup classifier trained successfully")
#         except Exception as e:
#             logger.error(f"Failed to setup backup classifier: {e}")
    
#     def detect_text_phishing(self, text: str) -> Dict[str, Any]:
#         """Enhanced phishing detection with multiple layers."""
#         start_time = time.time()
        
#         if not text or len(text.strip()) < 3:
#             return {
#                 'label': 'Invalid',
#                 'confidence': 0.0,
#                 'safety_percentage': 0.0,
#                 'features': [],
#                 'processing_time': time.time() - start_time
#             }
        
#         try:
#             # Always use heuristic analysis for better accuracy
#             heuristic_result = self._detect_with_heuristics(text, start_time)
            
#             # If heuristic detects high risk, trust it
#             if heuristic_result['safety_percentage'] < 60:
#                 return heuristic_result
            
#             # Use ML models for borderline cases
#             if self.backup_classifier:
#                 ml_result = self._detect_with_backup(text, start_time)
                
#                 # Combine results - take the more suspicious one
#                 if ml_result['safety_percentage'] < heuristic_result['safety_percentage']:
#                     return ml_result
#                 else:
#                     return heuristic_result
#             else:
#                 return heuristic_result
                
#         except Exception as e:
#             logger.error(f"Error in text detection: {e}")
#             return self._detect_with_heuristics(text, start_time)
    
#     def _detect_with_backup(self, text: str, start_time: float) -> Dict[str, Any]:
#         """Use backup classifier for detection."""
#         text_tfidf = self.tfidf_vectorizer.transform([text])
#         phishing_prob = self.backup_classifier.predict_proba(text_tfidf)[0][1]
        
#         confidence = phishing_prob * 100
#         safety_percentage = (1 - phishing_prob) * 100
        
#         if phishing_prob > 0.5:
#             label = 'Phishing'
#         elif phishing_prob > 0.3:
#             label = 'Suspicious'
#         else:
#             label = 'Safe'
        
#         features = self._extract_text_features(text)
        
#         return {
#             'label': label,
#             'confidence': confidence,
#             'safety_percentage': safety_percentage,
#             'features': features,
#             'processing_time': time.time() - start_time
#         }
    
#     def _detect_with_heuristics(self, text: str, start_time: float) -> Dict[str, Any]:
#         """Enhanced heuristic detection method."""
#         text_lower = text.lower()
        
#         # Enhanced suspicious keywords
#         urgent_keywords = ['urgent', 'immediately', 'asap', 'expire', 'deadline', 'limited time', 'act now', 'hurry']
#         security_keywords = ['verify', 'account', 'suspended', 'locked', 'security', 'alert', 'warning', 'compromised']
#         financial_keywords = ['bank', 'paypal', 'credit card', 'payment', 'refund', 'irs', 'tax', 'prize', 'winner', 'lottery']
#         action_keywords = ['click', 'download', 'install', 'call', 'contact', 'respond', 'confirm', 'update']
#         brand_keywords = ['amazon', 'apple', 'microsoft', 'google', 'facebook', 'netflix', 'ebay']
        
#         # Count keyword occurrences
#         urgent_count = sum(1 for word in urgent_keywords if word in text_lower)
#         security_count = sum(1 for word in security_keywords if word in text_lower)
#         financial_count = sum(1 for word in financial_keywords if word in text_lower)
#         action_count = sum(1 for word in action_keywords if word in text_lower)
#         brand_count = sum(1 for word in brand_keywords if word in text_lower)
        
#         # Calculate risk score
#         risk_score = 0
#         factors = []
        
#         # Urgency indicators (high weight)
#         if urgent_count > 0:
#             urgency_score = min(urgent_count * 0.3, 0.4)
#             risk_score += urgency_score
#             factors.append(f"Urgency indicators: {urgent_count}")
        
#         # Security-related keywords (high weight)
#         if security_count > 0:
#             security_score = min(security_count * 0.25, 0.3)
#             risk_score += security_score
#             factors.append(f"Security keywords: {security_count}")
        
#         # Financial keywords (medium weight)
#         if financial_count > 0:
#             financial_score = min(financial_count * 0.2, 0.25)
#             risk_score += financial_score
#             factors.append(f"Financial keywords: {financial_count}")
        
#         # Action keywords (medium weight)
#         if action_count > 0:
#             action_score = min(action_count * 0.15, 0.2)
#             risk_score += action_score
#             factors.append(f"Action keywords: {action_count}")
        
#         # Brand impersonation (high weight)
#         if brand_count > 0:
#             brand_score = min(brand_count * 0.2, 0.25)
#             risk_score += brand_score
#             factors.append(f"Brand mentions: {brand_count}")
        
#         # URL analysis (high weight)
#         url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#         urls = re.findall(url_pattern, text)
#         if urls:
#             suspicious_url_indicators = ['bit.ly', 'tinyurl', 'secure', 'verification', 'account', 'login', 'verify']
#             suspicious_urls = 0
#             for url in urls:
#                 if any(indicator in url.lower() for indicator in suspicious_url_indicators):
#                     suspicious_urls += 1
            
#             if suspicious_urls > 0:
#                 risk_score += min(suspicious_urls * 0.3, 0.4)
#                 factors.append(f"Suspicious URLs: {suspicious_urls}/{len(urls)}")
        
#         # Email patterns
#         email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
#         emails = re.findall(email_pattern, text)
#         if emails:
#             suspicious_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
#             official_domains = [email.split('@')[1].lower() for email in emails]
#             suspicious_email_count = sum(1 for domain in official_domains if domain in suspicious_domains)
            
#             if suspicious_email_count > 0 and (brand_count > 0 or security_count > 0):
#                 risk_score += 0.15
#                 factors.append("Suspicious email domain with brand/security claims")
        
#         # Phone number patterns
#         phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
#         phones = re.findall(phone_pattern, text)
#         if phones and (urgent_count > 0 or security_count > 0):
#             risk_score += 0.1
#             factors.append("Phone number with urgent/security language")
        
#         # Grammar and formatting issues
#         caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
#         if caps_ratio > 0.3:
#             risk_score += 0.1
#             factors.append("Excessive capitalization")
        
#         # Multiple exclamation marks
#         exclamation_count = text.count('!')
#         if exclamation_count > 2:
#             risk_score += 0.05
#             factors.append(f"Multiple exclamation marks: {exclamation_count}")
        
#         # Calculate final scores
#         risk_score = min(risk_score, 1.0)  # Cap at 1.0
#         confidence = risk_score * 100
#         safety_percentage = (1 - risk_score) * 100
        
#         # Determine label based on risk score
#         if risk_score > 0.6:
#             label = 'Phishing'
#         elif risk_score > 0.3:
#             label = 'Suspicious'
#         else:
#             label = 'Safe'
        
#         return {
#             'label': label,
#             'confidence': confidence,
#             'safety_percentage': safety_percentage,
#             'features': factors,
#             'processing_time': time.time() - start_time
#         }
    
#     def _extract_text_features(self, text: str) -> List[str]:
#         """Extract detailed features from text for analysis."""
#         features = []
#         text_lower = text.lower()
        
#         # Check for various suspicious patterns
#         patterns = {
#             'urgent_words': ['urgent', 'immediate', 'asap', 'expire', 'deadline'],
#             'security_words': ['verify', 'account', 'suspended', 'security', 'alert'],
#             'financial_words': ['bank', 'paypal', 'credit', 'payment', 'money'],
#             'action_words': ['click', 'download', 'call', 'contact', 'respond'],
#             'brands': ['amazon', 'apple', 'microsoft', 'google', 'facebook']
#         }
        
#         for category, words in patterns.items():
#             found = [word for word in words if word in text_lower]
#             if found:
#                 features.append(f"{category.replace('_', ' ').title()}: {', '.join(found)}")
        
#         # Check for URLs
#         url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#         urls = re.findall(url_pattern, text)
#         if urls:
#             features.append(f"URLs found: {len(urls)}")
        
#         # Check text characteristics
#         if len(text) > 1000:
#             features.append("Long text content")
#         elif len(text) < 50:
#             features.append("Very short text content")
        
#         # Check for excessive capitalization
#         caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
#         if caps_ratio > 0.3:
#             features.append(f"High capitalization ratio: {caps_ratio:.1%}")
        
#         return features

#     def extract_text_from_image(self, image_path: str) -> str:
#         """Extract text from image using OCR with enhanced error handling."""
#         if not OCR_AVAILABLE:
#             logger.warning("OCR not available - please install Tesseract")
#             return ""
        
#         try:
#             # Test if Tesseract is accessible
#             import subprocess
#             result = subprocess.run(['tesseract', '--version'], 
#                                  capture_output=True, text=True, timeout=10)
#             if result.returncode != 0:
#                 logger.error("Tesseract is not properly installed or not in PATH")
#                 return ""
            
#             # Open and process image
#             image = Image.open(image_path)
            
#             # Convert to RGB if necessary
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
            
#             # Extract text with specific configuration
#             custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?@#$%&*()_+-=[]{}|;:\'\"<>/\\ '
#             text = pytesseract.image_to_string(image, config=custom_config)
            
#             logger.info(f"OCR extracted {len(text)} characters from image")
#             return text.strip()
            
#         except subprocess.TimeoutExpired:
#             logger.error("Tesseract command timed out")
#             return ""
#         except Exception as e:
#             logger.error(f"Error extracting text from image: {e}")
#             return ""
    
#     def transcribe_audio(self, audio_path: str) -> str:
#         """Transcribe audio to text using Whisper."""
#         if not WHISPER_AVAILABLE or not self.whisper_model:
#             logger.warning("Whisper not available")
#             return ""
        
#         try:
#             result = self.whisper_model.transcribe(audio_path)
#             transcribed_text = result.get('text', '').strip()
#             logger.info(f"Audio transcribed: {len(transcribed_text)} characters")
#             return transcribed_text
#         except Exception as e:
#             logger.error(f"Error transcribing audio: {e}")
#             return ""
    
#     def extract_audio_from_video(self, video_path: str) -> Optional[str]:
#         """Extract audio from video file."""
#         if not MEDIA_PROCESSING_AVAILABLE:
#             logger.warning("Media processing libraries not available")
#             return None
        
#         try:
#             # For now, return the video path itself
#             # In production, you'd extract audio to a separate file
#             return video_path
#         except Exception as e:
#             logger.error(f"Error extracting audio from video: {e}")
#             return None
    
#     def analyze_file(self, file_path: str, file_type: str) -> Dict[str, Any]:
#         """Analyze uploaded file based on type with enhanced error handling."""
#         start_time = time.time()
        
#         try:
#             if file_type == 'text':
#                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                     content = f.read()
#                 return self.detect_text_phishing(content)
            
#             elif file_type == 'image':
#                 extracted_text = self.extract_text_from_image(file_path)
#                 if extracted_text and len(extracted_text.strip()) > 5:
#                     result = self.detect_text_phishing(extracted_text)
#                     result['extracted_text'] = extracted_text
#                     return result
#                 else:
#                     return {
#                         'label': 'Unable to analyze',
#                         'confidence': 0.0,
#                         'safety_percentage': 50.0,
#                         'features': ['No readable text detected in image - please ensure image is clear and contains text'],
#                         'processing_time': time.time() - start_time
#                     }
            
#             elif file_type == 'audio':
#                 transcribed_text = self.transcribe_audio(file_path)
#                 if transcribed_text and len(transcribed_text.strip()) > 5:
#                     result = self.detect_text_phishing(transcribed_text)
#                     result['transcribed_text'] = transcribed_text
#                     return result
#                 else:
#                     return {
#                         'label': 'Unable to analyze',
#                         'confidence': 0.0,
#                         'safety_percentage': 50.0,
#                         'features': ['No speech detected in audio file'],
#                         'processing_time': time.time() - start_time
#                     }
            
#             elif file_type == 'video':
#                 # Try to extract audio and transcribe
#                 audio_path = self.extract_audio_from_video(file_path)
#                 if audio_path:
#                     transcribed_text = self.transcribe_audio(audio_path)
#                     if transcribed_text and len(transcribed_text.strip()) > 5:
#                         result = self.detect_text_phishing(transcribed_text)
#                         result['transcribed_text'] = transcribed_text
#                         return result
                
#                 return {
#                     'label': 'Unable to analyze',
#                     'confidence': 0.0,
#                     'safety_percentage': 50.0,
#                     'features': ['No analyzable speech content detected in video'],
#                     'processing_time': time.time() - start_time
#                 }
            
#             else:
#                 return {
#                     'label': 'Unsupported format',
#                     'confidence': 0.0,
#                     'safety_percentage': 0.0,
#                     'features': ['Unsupported file format'],
#                     'processing_time': time.time() - start_time
#                 }
        
#         except Exception as e:
#             logger.error(f"Error analyzing file: {e}")
#             return {
#                 'label': 'Error',
#                 'confidence': 0.0,
#                 'safety_percentage': 0.0,
#                 'features': [f'Analysis error: {str(e)}'],
#                 'processing_time': time.time() - start_time
#             }

# # Helper functions remain the same
# def get_file_type(filename: str) -> str:
#     """Determine file type from filename."""
#     ext = filename.lower().split('.')[-1]
    
#     text_exts = {'txt', 'csv', 'json', 'log'}
#     image_exts = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}
#     audio_exts = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac'}
#     video_exts = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
    
#     if ext in text_exts:
#         return 'text'
#     elif ext in image_exts:
#         return 'image'
#     elif ext in audio_exts:
#         return 'audio'
#     elif ext in video_exts:
#         return 'video'
#     else:
#         return 'unknown'

# def allowed_file(filename: str) -> bool:
#     """Check if file extension is allowed."""
#     return '.' in filename and get_file_type(filename) != 'unknown'

# def secure_filename_custom(filename: str) -> str:
#     """Create secure filename with timestamp."""
#     timestamp = str(int(time.time()))
#     name, ext = os.path.splitext(filename)
#     name = re.sub(r'[^\w\s-]', '', name).strip()
#     name = re.sub(r'[-\s]+', '-', name)
#     return f"{timestamp}_{name}{ext}"

# def get_file_hash(file_path: str) -> str:
#     """Generate MD5 hash of file."""
#     hash_md5 = hashlib.md5()
#     try:
#         with open(file_path, "rb") as f:
#             for chunk in iter(lambda: f.read(4096), b""):
#                 hash_md5.update(chunk)
#         return hash_md5.hexdigest()
#     except Exception as e:
#         logger.error(f"Error generating file hash: {e}")
#         return ""

# # Global detector instance
# detector = PhishingDetector()

# def analyze_text(text: str) -> Dict[str, Any]:
#     """Main function to analyze text for phishing."""
#     return detector.detect_text_phishing(text)

# def analyze_file_content(file_path: str, filename: str) -> Dict[str, Any]:
#     """Main function to analyze file content."""
#     file_type = get_file_type(filename)
#     return detector.analyze_file(file_path, file_type)






import os
import re
import time
import logging
import hashlib
import subprocess
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for Tesseract installation
def check_tesseract_installation():
    """Check if Tesseract is properly installed."""
    try:
        # Try to find tesseract executable
        tesseract_path = shutil.which('tesseract')
        if tesseract_path:
            print(f"✅ Found Tesseract in PATH: {tesseract_path}")
            return tesseract_path
        
        # Try common Windows locations
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found Tesseract at: {path}")
                return path
        
        print("❌ Tesseract not found!")
        print("📋 To install Tesseract:")
        print("   1. Open Command Prompt as Administrator")
        print("   2. Run: winget install UB-Mannheim.TesseractOCR")
        print("   3. Restart your computer")
        print("   4. Or download from: https://github.com/UB-Mannheim/tesseract/wiki")
        return None
        
    except Exception as e:
        print(f"❌ Error checking Tesseract: {e}")
        return None

# Initialize OCR
tesseract_exe = check_tesseract_installation()
OCR_AVAILABLE = tesseract_exe is not None

if OCR_AVAILABLE:
    try:
        import pytesseract
        from PIL import Image
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe
        
        # Test if it works
        result = subprocess.run([tesseract_exe, '--version'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0] if result.stdout else 'unknown'
            print(f"✅ Tesseract is working: {version}")
        else:
            print(f"❌ Tesseract test failed: {result.stderr}")
            OCR_AVAILABLE = False
    except ImportError:
        print("❌ pytesseract or PIL not installed")
        OCR_AVAILABLE = False
    except Exception as e:
        print(f"❌ Tesseract setup error: {e}")
        OCR_AVAILABLE = False

# Audio processing
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper available for audio processing")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper not available")

class PhishingDetector:
    """Simple phishing detector with clear error messages."""
    
    def __init__(self):
        self.whisper_model = None
        self.load_models()
    
    def load_models(self):
        """Load audio models."""
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("tiny")
                logger.info("Loaded Whisper model")
            except Exception as e:
                logger.warning(f"Could not load Whisper model: {e}")
    
    def detect_text_phishing(self, text: str) -> Dict[str, Any]:
        """Enhanced phishing detection using heuristics."""
        start_time = time.time()
        
        if not text or len(text.strip()) < 3:
            return {
                'label': 'Invalid',
                'confidence': 0.0,
                'safety_percentage': 0.0,
                'features': ['Text too short to analyze'],
                'processing_time': time.time() - start_time
            }
        
        text_lower = text.lower()
        
        # Enhanced suspicious keywords
        urgent_keywords = ['urgent', 'immediately', 'asap', 'expire', 'deadline', 'limited time', 'act now', 'hurry']
        security_keywords = ['verify', 'account', 'suspended', 'locked', 'security', 'alert', 'warning', 'compromised']
        financial_keywords = ['bank', 'paypal', 'credit card', 'payment', 'refund', 'irs', 'tax', 'prize', 'winner', 'lottery']
        action_keywords = ['click', 'download', 'install', 'call', 'contact', 'respond', 'confirm', 'update']
        brand_keywords = ['amazon', 'apple', 'microsoft', 'google', 'facebook', 'netflix', 'paypal']
        
        # Count occurrences
        urgent_count = sum(1 for word in urgent_keywords if word in text_lower)
        security_count = sum(1 for word in security_keywords if word in text_lower)
        financial_count = sum(1 for word in financial_keywords if word in text_lower)
        action_count = sum(1 for word in action_keywords if word in text_lower)
        brand_count = sum(1 for word in brand_keywords if word in text_lower)
        
        # Calculate risk score
        risk_score = 0
        factors = []
        
        # High risk indicators
        if urgent_count > 0:
            risk_score += min(urgent_count * 0.3, 0.5)
            factors.append(f"Urgency words: {urgent_count}")
        
        if security_count > 0:
            risk_score += min(security_count * 0.3, 0.4)
            factors.append(f"Security alerts: {security_count}")
        
        if financial_count > 0:
            risk_score += min(financial_count * 0.25, 0.3)
            factors.append(f"Financial terms: {financial_count}")
        
        if action_count > 0:
            risk_score += min(action_count * 0.2, 0.25)
            factors.append(f"Action requests: {action_count}")
        
        if brand_count > 0:
            risk_score += min(brand_count * 0.2, 0.25)
            factors.append(f"Brand mentions: {brand_count}")
        
        # URL analysis
        url_pattern = r'http[s]?://[^\s]+'
        urls = re.findall(url_pattern, text)
        if urls:
            suspicious_indicators = ['bit.ly', 'tinyurl', 'secure', 'verification', 'login', 'account']
            suspicious_urls = sum(1 for url in urls if any(ind in url.lower() for ind in suspicious_indicators))
            if suspicious_urls > 0:
                risk_score += 0.3
                factors.append(f"Suspicious URLs: {suspicious_urls}/{len(urls)}")
        
        # Grammar indicators
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            risk_score += 0.1
            factors.append(f"High caps ratio: {caps_ratio:.1%}")
        
        exclamation_count = text.count('!')
        if exclamation_count > 2:
            risk_score += 0.05
            factors.append(f"Multiple exclamations: {exclamation_count}")
        
        # Final calculation
        risk_score = min(risk_score, 1.0)
        confidence = risk_score * 100
        safety_percentage = (1 - risk_score) * 100
        
        if risk_score > 0.6:
            label = 'Phishing'
        elif risk_score > 0.3:
            label = 'Suspicious'
        else:
            label = 'Safe'
        
        return {
            'label': label,
            'confidence': confidence,
            'safety_percentage': safety_percentage,
            'features': factors,
            'processing_time': time.time() - start_time
        }
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image with clear error messages."""
        if not OCR_AVAILABLE:
            logger.error("❌ Tesseract OCR not available")
            return ""
        
        try:
            if not os.path.exists(image_path):
                logger.error(f"❌ Image file not found: {image_path}")
                return ""
            
            logger.info(f"📷 Processing image: {image_path}")
            
            # Open image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text
            text = pytesseract.image_to_string(image, config='--psm 6')
            extracted_length = len(text.strip())
            
            logger.info(f"✅ Extracted {extracted_length} characters from image")
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ OCR Error: {e}")
            return ""
    
    def analyze_file(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Analyze file with better error handling."""
        start_time = time.time()
        
        try:
            if not os.path.exists(file_path):
                return {
                    'label': 'Error',
                    'confidence': 0.0,
                    'safety_percentage': 0.0,
                    'features': [f'File not found: {file_path}'],
                    'processing_time': time.time() - start_time
                }
            
            if file_type == 'text':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return self.detect_text_phishing(content)
            
            elif file_type == 'image':
                if not OCR_AVAILABLE:
                    return {
                        'label': 'Unable to analyze',
                        'confidence': 0.0,
                        'safety_percentage': 50.0,
                        'features': [
                            '❌ Tesseract OCR is not installed!',
                            '📋 To fix this:',
                            '   1. Open Command Prompt as Administrator',
                            '   2. Run: winget install UB-Mannheim.TesseractOCR',
                            '   3. Restart your computer',
                            '   4. Restart this application',
                            '',
                            'OR download from: https://github.com/UB-Mannheim/tesseract/wiki'
                        ],
                        'processing_time': time.time() - start_time
                    }
                
                extracted_text = self.extract_text_from_image(file_path)
                if extracted_text and len(extracted_text.strip()) > 3:
                    result = self.detect_text_phishing(extracted_text)
                    result['extracted_text'] = extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
                    return result
                else:
                    return {
                        'label': 'Unable to analyze',
                        'confidence': 0.0,
                        'safety_percentage': 50.0,
                        'features': [
                            'No readable text found in image',
                            'Image may be too blurry or contain no text',
                            'Try uploading a clearer image with visible text'
                        ],
                        'processing_time': time.time() - start_time
                    }
            
            else:
                return {
                    'label': 'Unsupported',
                    'confidence': 0.0,
                    'safety_percentage': 50.0,
                    'features': [f'File type "{file_type}" analysis not implemented yet'],
                    'processing_time': time.time() - start_time
                }
        
        except Exception as e:
            logger.error(f"❌ File analysis error: {e}")
            return {
                'label': 'Error',
                'confidence': 0.0,
                'safety_percentage': 0.0,
                'features': [f'Analysis error: {str(e)}'],
                'processing_time': time.time() - start_time
            }

# Helper functions
def get_file_type(filename: str) -> str:
    """Determine file type from filename."""
    ext = filename.lower().split('.')[-1]
    
    text_exts = {'txt', 'csv', 'json'}
    image_exts = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}
    audio_exts = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
    video_exts = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    
    if ext in text_exts:
        return 'text'
    elif ext in image_exts:
        return 'image'
    elif ext in audio_exts:
        return 'audio'
    elif ext in video_exts:
        return 'video'
    else:
        return 'unknown'

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and get_file_type(filename) != 'unknown'

def secure_filename_custom(filename: str) -> str:
    """Create secure filename with timestamp."""
    timestamp = str(int(time.time()))
    name, ext = os.path.splitext(filename)
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[-\s]+', '-', name)
    return f"{timestamp}_{name}{ext}"

def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash of file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return ""

# Global detector instance
detector = PhishingDetector()

def analyze_text(text: str) -> Dict[str, Any]:
    """Main function to analyze text for phishing."""
    return detector.detect_text_phishing(text)

def analyze_file_content(file_path: str, filename: str) -> Dict[str, Any]:
    """Main function to analyze file content."""
    file_type = get_file_type(filename)
    return detector.analyze_file(file_path, file_type)
