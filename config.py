"""
Configuration module for Human Voice Detection System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the voice detection system"""
    
    # HuggingFace Settings
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')
    
    # Model Settings
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
    VAD_MODEL = "pyannote/voice-activity-detection"
    
    # Audio Processing Settings
    SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', 16000))
    OUTPUT_FORMAT = os.getenv('OUTPUT_FORMAT', 'wav')
    
    # Speaker Settings
    MIN_SPEAKERS = int(os.getenv('MIN_SPEAKERS', 1))
    MAX_SPEAKERS = int(os.getenv('MAX_SPEAKERS', 10))
    
    # Output Settings
    OUTPUT_DIR = 'output'
    TEMP_DIR = 'temp'
    
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        if not cls.HUGGINGFACE_TOKEN:
            raise ValueError(
                "HuggingFace token is required. "
                "Please set HUGGINGFACE_TOKEN in .env file. "
                "Get your token from: https://huggingface.co/settings/tokens"
            )
        return True
