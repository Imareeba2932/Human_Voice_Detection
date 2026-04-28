"""
Audio/Video File Processor Module
Handles loading and preprocessing of audio/video files
"""
import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional
import librosa
import soundfile as sf
import numpy as np
from moviepy import VideoFileClip
from rich.console import Console

from .utils import get_file_extension, ensure_dir

console = Console()

class AudioProcessor:
    """Process audio and video files for voice detection"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize AudioProcessor
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        console.print(f"[cyan]Loading audio file: {filepath}[/cyan]")
        
        try:
            audio, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
            console.print(f"[green]✓ Audio loaded: {len(audio)/sr:.2f} seconds[/green]")
            return audio, sr
        except Exception as e:
            console.print(f"[red]✗ Error loading audio: {str(e)}[/red]")
            raise
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            output_path: Optional output path for extracted audio
            
        Returns:
            Path to extracted audio file
        """
        console.print(f"[cyan]Extracting audio from video: {video_path}[/cyan]")
        
        try:
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                video_name = Path(video_path).stem
                output_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
            
            # Ensure output directory exists
            ensure_dir(Path(output_path).parent)
            
            # Extract audio
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                raise ValueError("Video file has no audio track")
            
            video.audio.write_audiofile(
                output_path,
                fps=self.sample_rate,
                nbytes=2,
                codec='pcm_s16le',
                logger=None  # Suppress moviepy output
            )
            
            video.close()
            
            console.print(f"[green]✓ Audio extracted to: {output_path}[/green]")
            return output_path
            
        except Exception as e:
            console.print(f"[red]✗ Error extracting audio: {str(e)}[/red]")
            raise
    
    def process_file(self, filepath: str) -> Tuple[str, np.ndarray, int]:
        """
        Process audio or video file
        
        Args:
            filepath: Path to audio or video file
            
        Returns:
            Tuple of (audio_path, audio_data, sample_rate)
        """
        ext = get_file_extension(filepath)
        video_formats = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'mpeg', 'mpg']
        
        # Extract audio from video if needed
        if ext in video_formats:
            audio_path = self.extract_audio_from_video(filepath)
        else:
            audio_path = filepath
        
        # Load audio data
        audio_data, sr = self.load_audio(audio_path)
        
        return audio_path, audio_data, sr
    
    def save_audio(self, audio: np.ndarray, output_path: str, sample_rate: Optional[int] = None):
        """
        Save audio data to file
        
        Args:
            audio: Audio data as numpy array
            output_path: Output file path
            sample_rate: Sample rate (uses default if not specified)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure output directory exists
        ensure_dir(Path(output_path).parent)
        
        try:
            sf.write(output_path, audio, sample_rate)
            console.print(f"[green]✓ Audio saved to: {output_path}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Error saving audio: {str(e)}[/red]")
            raise
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to -1 to 1 range
        
        Args:
            audio: Audio data
            
        Returns:
            Normalized audio data
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio
