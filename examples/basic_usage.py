"""
Example: Basic usage of the Voice Detection System
"""
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from src.voice_detector import VoiceDetector
from src.speaker_diarization import SpeakerDiarizer
from src.audio_separator import AudioSeparator
from config import Config


def process_audio_file(input_file: str, output_dir: str = "output"):
    """
    Process an audio file to detect and separate speakers
    
    Args:
        input_file: Path to audio or video file
        output_dir: Output directory for results
    """
    print(f"Processing: {input_file}")
    
    # Validate configuration
    Config.validate()
    
    # Initialize components
    audio_processor = AudioProcessor(sample_rate=Config.SAMPLE_RATE)
    voice_detector = VoiceDetector(auth_token=Config.HUGGINGFACE_TOKEN)
    diarizer = SpeakerDiarizer(auth_token=Config.HUGGINGFACE_TOKEN)
    separator = AudioSeparator(sample_rate=Config.SAMPLE_RATE)
    
    # Process file
    audio_path, audio_data, sample_rate = audio_processor.process_file(input_file)
    
    # Detect voice
    vad_result = voice_detector.detect_voice(audio_path)
    voice_segments = voice_detector.get_voice_segments(vad_result)
    print(f"Found {len(voice_segments)} voice segments")
    
    # Diarize speakers
    diarization = diarizer.diarize(audio_path)
    speaker_segments = diarizer.get_speaker_segments(diarization)
    timeline = diarizer.get_timeline(diarization)
    
    print(f"Detected {len(speaker_segments)} speakers")
    
    # Print summary
    diarizer.print_summary(diarization)
    
    # Separate speakers
    output_files = separator.separate_speakers(
        audio_path,
        speaker_segments,
        output_dir,
        base_name="speaker"
    )
    
    # Save timeline
    timeline_path = Path(output_dir) / "timeline.json"
    diarizer.save_timeline(timeline, str(timeline_path))
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Speaker files: {list(output_files.values())}")
    print(f"Timeline: {timeline_path}")


if __name__ == "__main__":
    # Example usage
    # Replace with your actual audio/video file path
    input_file = "path/to/your/audio.wav"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    process_audio_file(input_file)
