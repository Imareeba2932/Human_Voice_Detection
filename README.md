# Human Voice Detection and Recognition System

A powerful Python-based system that detects human voices in audio/video files and separates different speakers using advanced AI models.

## Features

✅ **Voice Activity Detection (VAD)** - Detect human voice vs silence/noise/music  
✅ **Speaker Diarization** - Identify "who spoke when"  
✅ **Speaker Separation** - Extract individual speaker audio tracks  
✅ **Video Support** - Automatically extract audio from video files  
✅ **Multiple Formats** - Support for WAV, MP3, MP4, AVI, MKV, and more  
✅ **Timeline Export** - JSON timeline with speaker segments and timestamps  

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will install PyTorch, which may take some time. If you have a GPU, you can install the CUDA version of PyTorch for better performance.

### 2. Get HuggingFace Token

1. Create a free account at [HuggingFace](https://huggingface.co/)
2. Go to [Settings > Tokens](https://huggingface.co/settings/tokens)
3. Create a new token (read access is sufficient)

### 3. Accept Model Licenses

You must accept the licenses for the AI models:

1. Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Click "Accept" on the license agreement
3. Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Click "Accept" on the license agreement

### 4. Configure Environment

```bash
# Copy example env file
copy .env.example .env

# Edit .env and add your HuggingFace token
# HUGGINGFACE_TOKEN=your_token_here
```

## Usage

### Command Line Interface

#### Process a file (detect and separate speakers)

```bash
python main.py process input.mp4
```

With custom output directory:

```bash
python main.py process input.mp3 --output ./my_output
```

Specify min/max speakers:

```bash
python main.py process input.wav --min-speakers 2 --max-speakers 5
```

Detect voice only (skip diarization):

```bash
python main.py process input.mp4 --detect-only
```

#### Quick voice check

```bash
python main.py check input.mp3
```

#### Display setup instructions

```bash
python main.py setup
```

### Python API

```python
from src.audio_processor import AudioProcessor
from src.speaker_diarization import SpeakerDiarizer
from src.audio_separator import AudioSeparator
from config import Config

# Initialize
Config.validate()
processor = AudioProcessor()
diarizer = SpeakerDiarizer(auth_token=Config.HUGGINGFACE_TOKEN)
separator = AudioSeparator()

# Process file
audio_path, audio_data, sr = processor.process_file("input.mp4")

# Diarize
diarization = diarizer.diarize(audio_path)
speaker_segments = diarizer.get_speaker_segments(diarization)

# Separate speakers
output_files = separator.separate_speakers(
    audio_path,
    speaker_segments,
    "output"
)
```

See `examples/basic_usage.py` for a complete example.

## Output Files

When you process a file, the system generates:

- **`speaker_1.wav`**, **`speaker_2.wav`**, etc. - Individual speaker audio files
- **`timeline.json`** - Complete timeline with speaker segments and timestamps

Example timeline.json:

```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.5,
    "end": 3.2,
    "duration": 2.7,
    "start_formatted": "00:00:00.500",
    "end_formatted": "00:00:03.200"
  },
  {
    "speaker": "SPEAKER_01",
    "start": 3.5,
    "end": 7.1,
    "duration": 3.6,
    "start_formatted": "00:00:03.500",
    "end_formatted": "00:00:07.100"
  }
]
```

## Supported Formats

**Audio:** WAV, MP3, FLAC, OGG, M4A, AAC  
**Video:** MP4, AVI, MKV, MOV, WMV, FLV, WEBM

## System Requirements

- **Python:** 3.8 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** ~2GB for models and dependencies
- **GPU:** Optional, but recommended for faster processing

## Troubleshooting

### "HuggingFace token is required"

Make sure you've:
1. Created a `.env` file (copy from `.env.example`)
2. Added your HuggingFace token to the `.env` file

### "You must accept the model license"

Visit the model pages and accept the licenses:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### "No module named 'pyannote'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### Slow processing

- Processing speed depends on audio length and CPU/GPU
- First run downloads models (~500MB) which may take time
- Consider using a GPU for faster processing

## Advanced Configuration

Edit `config.py` or `.env` to customize:

- `SAMPLE_RATE` - Audio sample rate (default: 16000)
- `MIN_SPEAKERS` - Minimum speakers to detect (default: 1)
- `MAX_SPEAKERS` - Maximum speakers to detect (default: 10)
- `OUTPUT_FORMAT` - Output audio format (default: wav)

## Examples

Check the `examples/` directory for more usage examples.

## License

This project uses the following open-source libraries:
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - MIT License
- [PyTorch](https://pytorch.org/) - BSD License
- [librosa](https://librosa.org/) - ISC License

## Credits

Built using the excellent [pyannote.audio](https://github.com/pyannote/pyannote-audio) library by Hervé Bredin.

## Support

For issues and questions, please check:
1. This README
2. The troubleshooting section
3. [pyannote.audio documentation](https://github.com/pyannote/pyannote-audio)
