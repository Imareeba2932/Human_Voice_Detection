"""
Audio Separator Module
Separates audio by speaker based on diarization results
"""
import numpy as np
import soundfile as sf
from pathlib import Path
from rich.console import Console
from typing import Dict, List

from .utils import ensure_dir, get_output_path

console = Console()


class AudioSeparator:
    """Separate audio into individual speaker tracks"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize Audio Separator
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file as numpy array
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio data as numpy array
        """
        import librosa
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio
    
    def extract_segments(self, audio: np.ndarray, segments: List[tuple]) -> np.ndarray:
        """
        Extract and concatenate audio segments
        
        Args:
            audio: Full audio data
            segments: List of (start, end) tuples in seconds
            
        Returns:
            Concatenated audio segments
        """
        extracted_audio = []
        
        for start, end in segments:
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            
            # Ensure indices are within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            segment = audio[start_sample:end_sample]
            extracted_audio.append(segment)
        
        if extracted_audio:
            return np.concatenate(extracted_audio)
        else:
            return np.array([])
    
    def separate_speakers(
        self,
        audio_input,
        speaker_segments: Dict[str, List[tuple]],
        output_dir: str,
        base_name: str = "speaker"
    ) -> Dict[str, str]:
        """
        Separate speakers into individual audio files
        
        Args:
            audio_input: Path to audio file or numpy array of audio data
            speaker_segments: Dictionary mapping speakers to their segments
            output_dir: Output directory for separated audio files
            base_name: Base name for output files
            
        Returns:
            Dictionary mapping speaker labels to output file paths
        """
        console.print(f"[cyan]Separating speakers into individual audio files...[/cyan]")
        
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # Load audio if path is provided
        if isinstance(audio_input, str):
            audio = self.load_audio(audio_input)
        else:
            audio = audio_input
        
        output_files = {}
        
        # Process each speaker
        for speaker_idx, (speaker, segments) in enumerate(speaker_segments.items(), 1):
            # Extract speaker's audio segments
            speaker_audio = self.extract_segments(audio, segments)
            
            if len(speaker_audio) == 0:
                console.print(f"[yellow]⚠ Warning: No audio for {speaker}[/yellow]")
                continue
            
            # Generate output filename
            output_filename = f"{base_name}_{speaker_idx}.wav"
            output_path = str(Path(output_dir) / output_filename)
            
            # Save audio
            sf.write(output_path, speaker_audio, self.sample_rate)
            
            output_files[speaker] = output_path
            
            duration = len(speaker_audio) / self.sample_rate
            console.print(f"[green]✓ {speaker}: {output_filename} ({duration:.2f}s)[/green]")
        
        return output_files
    
    def create_speaker_timeline_audio(
        self,
        audio_path: str,
        timeline: List[Dict],
        output_dir: str,
        silence_duration: float = 0.5
    ) -> str:
        """
        Create audio file with speakers in chronological order separated by silence
        
        Args:
            audio_path: Path to original audio file
            timeline: Timeline with speaker segments
            output_dir: Output directory
            silence_duration: Duration of silence between speakers (seconds)
            
        Returns:
            Path to output file
        """
        console.print(f"[cyan]Creating timeline audio with speaker separations...[/cyan]")
        
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # Load full audio
        audio = self.load_audio(audio_path)
        
        # Create silence
        silence_samples = int(silence_duration * self.sample_rate)
        silence = np.zeros(silence_samples)
        
        # Build timeline audio
        timeline_audio = []
        
        for segment in timeline:
            start = segment['start']
            end = segment['end']
            
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            
            # Extract segment
            segment_audio = audio[start_sample:end_sample]
            
            # Add segment and silence
            timeline_audio.append(segment_audio)
            timeline_audio.append(silence)
        
        # Concatenate all segments
        if timeline_audio:
            final_audio = np.concatenate(timeline_audio)
        else:
            final_audio = np.array([])
        
        # Save
        output_path = str(Path(output_dir) / "timeline_separated.wav")
        sf.write(output_path, final_audio, self.sample_rate)
        
        duration = len(final_audio) / self.sample_rate
        console.print(f"[green]✓ Timeline audio saved: {output_path} ({duration:.2f}s)[/green]")
        
        return output_path
