# Quick Start Guide

## Installation

1. **Install Python dependencies:**
   ```bash
   cd d:\Batches\2026\Human_Voice_Detection
   pip install -r requirements.txt
   ```

2. **Setup HuggingFace:**
   - Get token from: https://huggingface.co/settings/tokens
   - Accept licenses:
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0

3. **Configure environment:**
   ```bash
   copy .env.example .env
   ```
   Then edit `.env` and add your HuggingFace token

## Usage

### Display setup help:
```bash
python main.py setup
```

### Quick voice check:
```bash
python main.py check path/to/audio.mp3
```

### Process and separate speakers:
```bash
python main.py process path/to/audio.mp4
```

### With custom output:
```bash
python main.py process path/to/audio.mp3 --output ./my_output
```

## Testing

You can test with any audio or video file that contains multiple speakers, such as:
- Podcast episodes
- Interview recordings
- Conference calls
- YouTube videos (download audio first)

The system will automatically:
1. Extract audio (if video)
2. Detect voice activity
3. Identify speakers
4. Separate each speaker into individual WAV files
5. Generate a timeline JSON with timestamps
