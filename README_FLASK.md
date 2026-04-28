# VoiceSense AI - Flask Integration

This project has been integrated with Flask to provide a premium web interface for Human Voice Detection and Speaker Diarization.

## Prerequisites

1. **HuggingFace Token**: You need a token from HuggingFace to use the `pyannote.audio` models.
   - Get it at: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Accept Model Licenses**: You must accept the terms for these models on HuggingFace:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   - Create a `.env` file in the root directory (copy from `.env.example`).
   - Add your `HUGGINGFACE_TOKEN`.

## Running the Application

Start the Flask server:
```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

## Project Structure (Simple Architecture)

- `app.py`: Main Flask application containing all routes and backend orchestration.
- `templates/`: HTML templates with premium Glassmorphism UI.
- `static/`: CSS, JS, and uploaded/processed audio files.
- `src/`: Core logic modules (Audio Processing, VAD, Diarization).
- `config.py`: Configuration management.
