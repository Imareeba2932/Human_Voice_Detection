"""
Voice Activity Detection Module
Detects human voice segments in audio
"""
import torch
import numpy as np
from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import VoiceActivityDetection
from rich.console import Console

console = Console()


class VoiceDetector:
    """Detect voice activity in audio files"""
    
    def __init__(self, auth_token: str):
        """
        Initialize Voice Detector
        
        Args:
            auth_token: HuggingFace authentication token
        """
        self.auth_token = auth_token
        self.pipeline = None
        
    def load_model(self):
        """Load the voice activity detection model"""
        if self.pipeline is not None:
            return
        
        console.print("[cyan]Loading Voice Activity Detection model...[/cyan]")
        
        try:
            # Load pre-trained segmentation model
            model = Model.from_pretrained(
                "pyannote/segmentation-3.0",
                token=self.auth_token
            )
            
            # Create Voice Activity Detection pipeline
            self.pipeline = VoiceActivityDetection(segmentation=model)
            
            # Default hyper-parameters
            HYPER_PARAMETERS = {
                # remove speech regions shorter than that many seconds
                "min_duration_on": 0.1,
                # fill non-speech regions shorter than that many seconds
                "min_duration_off": 0.1,
            }
            self.pipeline.instantiate(HYPER_PARAMETERS)
            
            console.print("[green]✓ VAD model loaded successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Error loading VAD model: {str(e)}[/red]")
            raise
    
    def detect_voice(self, audio_data, audio_path: str = None):
        """
        Detect voice activity in audio file
        
        Args:
            audio_data: Audio data as dict {'waveform': tensor, 'sample_rate': int} or path
            audio_path: Path to audio file for logging
            
        Returns:
            Annotation object with voice segments
        """
        if self.pipeline is None:
            self.load_model()
        
        if audio_path is None:
            audio_path = "preloaded audio"
        
        console.print(f"[cyan]Detecting voice activity in: {audio_path}[/cyan]")
        
        try:
            # Ensure audio_data is in the format pyannote expects
            if isinstance(audio_data, dict) and 'waveform' in audio_data:
                waveform = audio_data['waveform']
                
                # Convert numpy to torch if needed
                if isinstance(waveform, np.ndarray):
                    waveform = torch.from_numpy(waveform).float()
                
                # Ensure it's (channel, time)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                
                audio_data['waveform'] = waveform
                
            # Run voice activity detection
            result = self.pipeline(audio_data)
            
            # Handle potential new return type (DiarizeOutput-like container)
            if hasattr(result, "voice_activity_detection"):
                vad_result = result.voice_activity_detection
            elif hasattr(result, "segmentation"):
                vad_result = result.segmentation
            else:
                vad_result = result
            
            # Count voice segments
            num_segments = len(list(vad_result.itertracks()))
            total_duration = sum(segment.end - segment.start for segment, _ in vad_result.itertracks())
            
            console.print(f"[green]✓ Voice detected: {num_segments} segments, {total_duration:.2f}s total duration[/green]")
            
            return vad_result
            
        except Exception as e:
            console.print(f"[red]✗ Error detecting voice: {str(e)}[/red]")
            raise
    
    def get_voice_segments(self, vad_result) -> list:
        """
        Extract voice segments as list of (start, end) tuples
        
        Args:
            vad_result: VAD annotation result
            
        Returns:
            List of tuples [(start_time, end_time), ...]
        """
        segments = []
        for segment, _ in vad_result.itertracks():
            segments.append((segment.start, segment.end))
        
        return segments
    
    def has_voice(self, audio_data, audio_path: str = None) -> bool:
        """
        Check if audio file contains any voice activity
        
        Args:
            audio_data: Audio data as dict {'waveform': tensor, 'sample_rate': int} or path
            audio_path: Path to audio file for logging
            
        Returns:
            True if voice is detected, False otherwise
        """
        vad_result = self.detect_voice(audio_data, audio_path)
        return len(list(vad_result.itertracks())) > 0
