import os
import uuid
import json
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Import local modules
from src.audio_processor import AudioProcessor
from src.voice_detector import VoiceDetector
from src.speaker_diarization import SpeakerDiarizer
from src.audio_separator import AudioSeparator
from src.utils import validate_audio_file, ensure_dir, format_timestamp
from config import Config

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Login Manager Configuration
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mpeg', 'mp4', 'avi', 'mkv', 'mov', 'webm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_exists = User.query.filter((User.username == username) | (User.email == email)).first()
        if user_exists:
            flash('Username or email already exists.')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Create a unique directory for this session
            session_id = str(uuid.uuid4())
            session_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
            session_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
            os.makedirs(session_upload_dir, exist_ok=True)
            os.makedirs(session_output_dir, exist_ok=True)
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_upload_dir, filename)
            file.save(file_path)
            
            # Process the file
            try:
                # Validate configuration
                Config.validate()
                
                # Initialize processors
                audio_processor = AudioProcessor(sample_rate=Config.SAMPLE_RATE)
                voice_detector = VoiceDetector(auth_token=Config.HUGGINGFACE_TOKEN)
                
                # Process audio/video file
                audio_path, audio_data, sample_rate = audio_processor.process_file(file_path)
                
                # Detect voice activity
                vad_result = voice_detector.detect_voice({'waveform': audio_data, 'sample_rate': sample_rate}, audio_path)
                voice_segments = voice_detector.get_voice_segments(vad_result)
                
                # Diarization
                diarizer = SpeakerDiarizer(
                    auth_token=Config.HUGGINGFACE_TOKEN,
                    min_speakers=Config.MIN_SPEAKERS,
                    max_speakers=Config.MAX_SPEAKERS
                )
                
                diarization = diarizer.diarize({'waveform': audio_data, 'sample_rate': sample_rate}, audio_path)
                
                # Get results
                timeline = diarizer.get_timeline(diarization)
                speaker_stats = diarizer.get_speaker_stats(diarization)
                speaker_segments = diarizer.get_speaker_segments(diarization)
                
                # Separate speakers
                separator = AudioSeparator(sample_rate=Config.SAMPLE_RATE)
                output_files = separator.separate_speakers(
                    audio_data,
                    speaker_segments,
                    session_output_dir,
                    base_name=f"speaker"
                )
                
                # Prepare data for template
                results = {
                    'session_id': session_id,
                    'filename': filename,
                    'total_segments': len(timeline),
                    'total_speakers': len(speaker_stats),
                    'timeline': timeline,
                    'speaker_stats': speaker_stats,
                    'output_files': {}
                }
                
                # Convert absolute paths to relative URLs for the web
                for speaker, path in output_files.items():
                    rel_path = os.path.relpath(path, 'static').replace('\\', '/')
                    results['output_files'][speaker] = rel_path
                
                # Save results to a JSON for persistence if needed
                with open(os.path.join(session_output_dir, 'results.json'), 'w') as f:
                    json.dump(results, f)
                
                return render_template('results.html', results=results)
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(url_for('index'))
                
        flash('File type not allowed')
        return redirect(url_for('index'))
    
    # Handle GET request
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
