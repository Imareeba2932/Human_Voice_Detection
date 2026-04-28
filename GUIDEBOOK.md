# Complete Guidebook: Human Voice Detection & Speaker Diarization System

Welcome to the comprehensive guide for building a state-of-the-art Voice Intelligence System. This project detects human voices, identifies distinct speakers, and separates their voices into individual audio tracks using advanced AI.

---

## Module 1: Project Overview & Environment Setup

### 1.1 Project Objective
The goal is to build a web application where users can upload audio or video files. The AI will:
1. Filter out background noise and identify human speech.
2. Identify different individuals speaking.
3. Provide a timeline of the conversation.
4. Allow downloading separate audio files for each speaker.

### 1.2 Folder Structure
Create the following structure in your project directory:
```text
/
├── app.py (Main Backend)
├── config.py (Settings)
├── .env (Credentials)
├── src/ (Logic Folder)
│   ├── __init__.py
│   ├── audio_processor.py
│   ├── voice_detector.py
│   ├── speaker_diarization.py
│   ├── audio_separator.py
│   └── utils.py
├── templates/ (UI Files)
│   ├── base.html
│   ├── index.html
│   └── results.html
├── static/ (Assets)
│   ├── css/
│   ├── js/
│   ├── uploads/
│   └── output/
└── requirements.txt
```

### 1.3 Requirements File
Create `requirements.txt` with all necessary libraries.

**File: `requirements.txt`**
```text
pyannote.audio>=3.1.0
torch>=2.0.0
torchaudio>=2.0.0
pydub>=0.25.1
librosa>=0.10.0
soundfile>=0.12.1
moviepy>=1.0.3
python-dotenv>=1.0.0
numpy>=1.24.0
scipy>=1.10.0
flask>=3.0.0
werkzeug>=3.0.0
rich>=13.0.0
click>=8.1.0
```

### Line-by-Line Explanation:
- `pyannote.audio`: The core AI library for voice detection and speaker identification.
- `torch` & `torchaudio`: Power the deep learning models used by pyannote.
- `pydub`, `librosa`, `soundfile`: Used for manipulating and reading audio data.
- `moviepy`: Used to extract audio from video files (MP4, AVI, etc.).
- `python-dotenv`: Loads secret keys (like API tokens) from a `.env` file.
- `flask`: The web framework used to build the beautiful user interface.
- `rich` & `click`: Provide colorful logging and command-line support.

### Running Instructions:
1. Open your terminal in the project folder.
2. Run: `pip install -r requirements.txt`

---

## Module 2: Configuration & Environment Variables

We need a safe way to store settings and API tokens.

**File: `config.py`**
```python
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the voice detection system"""
    
    # HuggingFace Settings (Required for AI models)
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')
    
    # Model Names
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
    VAD_MODEL = "pyannote/voice-activity-detection"
    
    # Audio Settings
    SAMPLE_RATE = 16000  # Standard rate for AI models
    
    # Speaker Range
    MIN_SPEAKERS = 1
    MAX_SPEAKERS = 10
    
    @classmethod
    def validate(cls):
        """Checks if the token is present"""
        if not cls.HUGGINGFACE_TOKEN:
            raise ValueError("HuggingFace token is missing in .env file!")
        return True
```

### Line-by-Line Explanation:
1-2: Import libraries for accessing the system and loading environment variables.
5: `load_dotenv()` reads the `.env` file so we don't hardcode sensitive keys.
7: `class Config` groups all our settings in one place.
11: `HUGGINGFACE_TOKEN` gets your secret key from the `.env`.
14-15: Specifies which AI models to download from HuggingFace.
18: `SAMPLE_RATE` is set to 16kHz because AI models are trained on this frequency.
24: `validate()` function ensures the app doesn't crash later by checking the token now.

### Running Instructions:
1. Create a file named `.env`.
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Copy your token and write it in `.env` like this: `HUGGINGFACE_TOKEN=your_token_here`.

---

## Module 3: Utility Functions

These are helper functions used throughout the project.

**File: `src/utils.py`**
```python
import os
from pathlib import Path
from datetime import timedelta

def ensure_dir(directory: str) -> Path:
    """Creates a folder if it doesn't exist"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def format_timestamp(seconds: float) -> str:
    """Converts 75.5 seconds into 00:01:15.500"""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = td.total_seconds() % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def get_file_extension(filepath: str) -> str:
    """Gets extension like 'mp3' or 'mp4'"""
    return Path(filepath).suffix.lower().strip('.')

def validate_audio_file(filepath: str):
    """Checks if the file is supported"""
    supported = ['wav', 'mp3', 'mp4', 'avi', 'mkv', 'webm']
    ext = get_file_extension(filepath)
    if ext in supported:
        return True, "Valid file"
    return False, f"Unsupported format: {ext}"
```

### Line-by-Line Explanation:
5: `ensure_dir` makes sure we have a place to save our results without errors.
12: `format_timestamp` takes a raw number of seconds and turns it into a readable time (HH:MM:SS) for the user.
21: `get_file_extension` helps the app identify if the user uploaded audio or video.
25: `validate_audio_file` acts as a security guard, only allowing file types the app can process.

### Running Instructions:
- Create the `src/` folder and save this file inside it.

---

## Module 4: Audio Processing Core

This module handles loading audio and extracting audio from video files.

**File: `src/audio_processor.py`**
```python
import os
import tempfile
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from moviepy import VideoFileClip
from .utils import get_file_extension, ensure_dir

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def process_file(self, filepath: str):
        """Converts video to audio if needed and loads it"""
        ext = get_file_extension(filepath)
        video_formats = ['mp4', 'avi', 'mkv', 'mov']
        
        if ext in video_formats:
            # Extract audio from video
            video = VideoFileClip(filepath)
            temp_audio = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
            video.audio.write_audiofile(temp_audio, fps=self.sample_rate, logger=None)
            filepath = temp_audio
            video.close()
            
        # Load audio data using librosa
        audio_data, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
        return filepath, audio_data, sr
```

### Line-by-Line Explanation:
10: `AudioProcessor` is a class designed to handle all sound-related tasks.
14: `process_file` is the main entrance for any file (audio or video).
17: It checks the extension. If it's a video (like MP4), it triggers extraction.
19: `VideoFileClip` opens the video file.
21: `write_audiofile` saves just the sound part as a `.wav` file.
26: `librosa.load` converts the audio file into a "numpy array" (a list of numbers) that the AI can understand.

### Running Instructions:
- Ensure `moviepy` is installed. This module will be tested once we reach the integration stage.

---

## Module 5: Voice Activity Detection (VAD)

This module uses AI to find where humans are speaking.

**File: `src/voice_detector.py`**
```python
import torch
import numpy as np
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

class VoiceDetector:
    def __init__(self, auth_token: str):
        self.auth_token = auth_token
        self.pipeline = None
        
    def load_model(self):
        """Downloads/Loads the VAD model from HuggingFace"""
        model = Model.from_pretrained("pyannote/segmentation-3.0", token=self.auth_token)
        self.pipeline = VoiceActivityDetection(segmentation=model)
        self.pipeline.instantiate({"min_duration_on": 0.1, "min_duration_off": 0.1})
        
    def detect_voice(self, audio_data):
        """Detects voice in the provided waveform"""
        if self.pipeline is None: self.load_model()
        
        # Prepare waveform for AI
        waveform = audio_data['waveform']
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        # Run detection
        result = self.pipeline({"waveform": waveform, "sample_rate": audio_data['sample_rate']})
        
        # Handle new return type in newer pyannote versions
        if hasattr(result, "voice_activity_detection"):
            vad_result = result.voice_activity_detection
        else:
            vad_result = result
            
        return vad_result

    def get_voice_segments(self, vad_result):
        """Returns list of (start, end) times"""
        return [(s.start, s.end) for s, _ in vad_result.itertracks()]
```

### Line-by-Line Explanation:
12: `Model.from_pretrained` connects to HuggingFace and downloads the AI brain.
14: `instantiate` sets the sensitivity of the detector (avoiding tiny noises).
16: `detect_voice` takes the audio numbers (waveform) and looks for patterns.
20-23: Converts the numbers into a format (Tensor) that Deep Learning models (PyTorch) require.
26: The actual AI magic happens here; `self.pipeline` returns where the speech is.
30: `get_voice_segments` cleans up the AI output into simple start/end times.

---

## Module 6: Speaker Diarization

This module identifies who is speaking (Speaker 1, Speaker 2, etc.).

**File: `src/speaker_diarization.py`**
```python
from pyannote.audio import Pipeline
import torch
import numpy as np
from .utils import format_timestamp

class SpeakerDiarizer:
    def __init__(self, auth_token: str, min_speakers=1, max_speakers=10):
        self.auth_token = auth_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.pipeline = None
        
    def load_model(self):
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=self.auth_token)
        
    def diarize(self, audio_data):
        if self.pipeline is None: self.load_model()
        
        waveform = audio_data['waveform']
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        # identify speakers
        result = self.pipeline(
            {"waveform": waveform, "sample_rate": audio_data['sample_rate']},
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers
        )
        
        # Handle new return type in pyannote.audio 3.1.0+
        if hasattr(result, "speaker_diarization"):
            diarization = result.speaker_diarization
        else:
            diarization = result
            
        return diarization

    def get_timeline(self, diarization):
        """Creates a list of who spoke when"""
        timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end,
                'start_formatted': format_timestamp(turn.start),
                'end_formatted': format_timestamp(turn.end),
                'duration': turn.end - turn.start
            })
        return timeline

    def get_speaker_stats(self, diarization):
        """Calculates total time for each speaker"""
        stats = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in stats:
                stats[speaker] = {'total_duration': 0, 'num_segments': 0}
            stats[speaker]['total_duration'] += (turn.end - turn.start)
            stats[speaker]['num_segments'] += 1
        return stats
```

### Line-by-Line Explanation:
12: Downloads the specific "Speaker Diarization" model.
15: `diarize` is the primary function to separate speakers.
24-28: It runs the AI pipeline, identifying different voice prints.
31: `get_timeline` builds the "chat history" of the audio file.
45: `get_speaker_stats` sums up how much each person spoke, which is great for analytics.

---

## Module 7: Audio track Separation

This module cuts the audio into small pieces based on who is speaking.

**File: `src/audio_separator.py`**
```python
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from .utils import ensure_dir

class AudioSeparator:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def separate_speakers(self, audio_data, speaker_segments, output_dir):
        """Saves a separate wav file for each speaker"""
        ensure_dir(output_dir)
        output_files = {}
        
        for speaker, segments in speaker_segments.items():
            speaker_audio = []
            for start, end in segments:
                # Calculate sample indices
                s_idx = int(start * self.sample_rate)
                e_idx = int(end * self.sample_rate)
                speaker_audio.append(audio_data[s_idx:e_idx])
            
            if speaker_audio:
                # Combine all segments for this speaker
                combined = np.concatenate(speaker_audio)
                file_path = os.path.join(output_dir, f"{speaker}.wav")
                sf.write(file_path, combined, self.sample_rate)
                output_files[speaker] = file_path
                
        return output_files
```

### Line-by-Line Explanation:
10: `separate_speakers` iterates through each identified person.
14: For each speaker, it finds all the time slots they were talking in.
18-19: It calculates exactly which "numbers" in the audio list belong to that speaker.
24: `np.concatenate` joins those pieces into one continuous voice file.
26: `sf.write` saves the final voice file to your computer.

---

## Module 8: Flask Backend (Part 1 - Setup)

Now we start building the web server.

**File: `app.py`**
```python
import os
import uuid
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from config import Config

# Initialize Flask
app = Flask(__name__)
app.secret_key = "secret_key_for_session"

# Setup Upload Folders
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Renders the Home Page"""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

### Line-by-Line Explanation:
9: `app = Flask(__name__)` tells Python to start a web server.
10: `secret_key` is needed for error messages (flashing).
13-16: We create folders to store uploaded files and processed voices.
18: `@app.route('/')` defines what happens when you visit the website address.
22: `render_template` looks for the UI file named `index.html`.

### Running Instructions:
1. Run `python app.py`.
2. It will show an error because `index.html` doesn't exist yet. We will fix this in the next module.

---

## Module 9: Premium UI - Base Template

We use **Glassmorphism** for a professional feel.

**File: `templates/base.html`**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VoiceSense AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root { --primary: #6366f1; --bg: #0f172a; --glass: rgba(255, 255, 255, 0.05); }
        body { font-family: 'Outfit', sans-serif; background: var(--bg); color: white; margin: 0; }
        .glass-card { background: var(--glass); backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 2rem; }
        .nav { padding: 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: center; }
        .logo { font-size: 1.5rem; font-weight: 700; color: var(--primary); text-decoration: none; }
    </style>
</head>
<body>
    <div class="nav"><a href="/" class="logo"><i class="fas fa-waveform"></i> VoiceSense AI</a></div>
    <div style="max-width: 1000px; margin: 3rem auto; padding: 0 1rem;">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
```

### Line-by-Line Explanation:
8: We are using "Outfit", a premium Google font.
11-13: UI Variables. `--bg` is a deep blue/black, and `--glass` is used for the modern semi-transparent look.
15: `.glass-card` uses `backdrop-filter: blur` to create that stunning blurry-glass effect.
23: `{% block content %}` is a placeholder where other pages will insert their code.

---

## Module 10: Premium UI - Home Page

The page where users upload files.

**File: `templates/index.html`**
```html
{% extends "base.html" %}
{% block content %}
<div class="glass-card" style="text-align: center;">
    <h1>Analyze Audio with AI</h1>
    <p style="color: #94a3b8; margin-bottom: 2rem;">Upload your audio or video file to identify speakers and separate voices.</p>
    
    <form action="/upload" method="post" enctype="multipart/form-data">
        <label for="file" style="display: block; border: 2px dashed #6366f1; padding: 4rem; border-radius: 20px; cursor: pointer; transition: 0.3s;">
            <i class="fas fa-cloud-upload-alt" style="font-size: 3rem; color: #6366f1;"></i>
            <p>Click to browse or drag and drop</p>
            <input type="file" name="file" id="file" hidden onchange="this.form.submit()">
        </label>
    </form>
</div>
{% endblock %}
```

### Line-by-Line Explanation:
1: `{% extends "base.html" %}` tells the app to use the layout we made in Module 9.
8: The `enctype="multipart/form-data"` is CRITICAL; without it, you can't upload files.
9: This label creates a big, beautiful dashed box for uploading.
12: `onchange="this.form.submit()"` automatically starts the processing as soon as you pick a file.

### Running Instructions:
1. Run `app.py`.
2. Visit `localhost:5000`. You will now see a beautiful upload screen!

---

## Module 11: Backend Integration (Full Processing)

Now we connect the AI logic to the web interface.

**UPDATE: `app.py`** (Replace existing content or add these routes)
```python
from src.audio_processor import AudioProcessor
from src.voice_detector import VoiceDetector
from src.speaker_diarization import SpeakerDiarizer
from src.audio_separator import AudioSeparator

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files: return redirect('/')
    file = request.files['file']
    if file.filename == '': return redirect('/')
    
    # Save the file
    session_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    
    # AI Processing
    proc = AudioProcessor()
    _, audio_data, sr = proc.process_file(save_path)
    
    # Diarization
    diarizer = SpeakerDiarizer(auth_token=Config.HUGGINGFACE_TOKEN)
    diarization = diarizer.diarize({'waveform': audio_data, 'sample_rate': sr})
    
    # Results
    timeline = diarizer.get_timeline(diarization)
    stats = diarizer.get_speaker_stats(diarization)
    
    # Separation
    sep = AudioSeparator()
    output_dir = os.path.join(OUTPUT_FOLDER, session_id)
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_segments: speaker_segments[speaker] = []
        speaker_segments[speaker].append((turn.start, turn.end))
        
    out_files = sep.separate_speakers(audio_data, speaker_segments, output_dir)
    web_files = {}
    for speaker, path in out_files.items():
        rel_path = os.path.relpath(path, 'static').replace('\\', '/')
        web_files[speaker] = rel_path

    return render_template('results.html', timeline=timeline, stats=stats, files=web_files)
```

### Line-by-Line Explanation:
8: `handle_upload` captures the file being sent from the UI.
13: `uuid.uuid4()` gives a unique ID to every user so files don't mix up.
18-20: Calls our Module 4 logic to convert and load the sound.
23: Calls Module 6 AI logic to find speakers.
31-35: Logic to prepare the data for our track-cutting function.
37: Calls Module 7 to save the separate voice files.

---

## Module 12: Premium UI - Results Page

Displays the conversation timeline and audio players.

**File: `templates/results.html`**
```html
{% extends "base.html" %}
{% block content %}
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
    <!-- Timeline -->
    <div class="glass-card">
        <h3>Conversation Timeline</h3>
        {% for item in timeline %}
        <div style="border-left: 2px solid #6366f1; padding-left: 1rem; margin-bottom: 1rem;">
            <b style="color: #6366f1;">{{ item.speaker }}</b> 
            <span style="font-size: 0.8rem; color: #94a3b8;">{{ item.start_formatted }}</span>
            <p style="margin: 0.5rem 0;">Spoke for {{ "%.2f"|format(item.duration) }}s</p>
        </div>
        {% endfor %}
    </div>

    <!-- Speaker Downloads -->
    <div class="glass-card">
        <h3>Separated Voices</h3>
        {% for speaker, file_url in files.items() %}
        <div style="margin-bottom: 1.5rem; background: rgba(255,255,255,0.02); padding: 1rem; border-radius: 12px;">
            <p><b>{{ speaker }}</b></p>
            <audio controls style="width: 100%;"><source src="{{ url_for('static', filename=file_url) }}" type="audio/wav"></audio>
        </div>
        {% endfor %}
    </div>
</div>
{% endblock %}
```

### Line-by-Line Explanation:
4: `grid-template-columns: 1fr 1fr` splits the screen into two nice halves.
9: `{% for item in timeline %}` loops through our chat history data.
23: `audio controls` adds a built-in player so you can listen to each person individually.

---

## Module 13: Styling Optimization

Final touch to make the app look "Apple-like" premium.

**File: `static/css/style.css`**
```css
.glass-card {
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    transition: transform 0.3s ease;
}
.glass-card:hover { transform: translateY(-5px); }
h1, h3 { letter-spacing: -0.02em; font-weight: 600; }
audio { filter: invert(0.8); }
```

### Line-by-Line Explanation:
2: Adds a soft shadow to the cards (makes them "pop").
5: Makes the card lift up slightly when you hover over it (interactive feel).
7: Inverts the standard white audio player to match our dark mode theme.

---

## Module 14: Final Run & Troubleshooting

### To Test the Whole Project:
1. Run `python app.py`.
2. Open `http://localhost:5000`.
3. Upload a file with 2-3 people talking.
4. Wait for processing (the timeline will appear).
5. Open the `.wav` files in `static/output` to see individual voices.

### Troubleshooting:
- **"Torch not found"**: Run `pip install torch`.
- **"HuggingFace license"**: Ensure you clicked "Accept License" on the model page at HuggingFace.

---

