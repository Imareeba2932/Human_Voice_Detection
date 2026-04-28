"""
Speaker Diarization Module
Identifies "who spoke when" in audio files
"""
from pyannote.audio import Pipeline
import torch
import numpy as np
from rich.console import Console
from typing import Dict, List
import json

from .utils import format_timestamp

console = Console()


class SpeakerDiarizer:
    """Perform speaker diarization to identify different speakers"""
    
    def __init__(self, auth_token: str, min_speakers: int = 1, max_speakers: int = 10):
        """
        Initialize Speaker Diarizer
        
        Args:
            auth_token: HuggingFace authentication token
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
        """
        self.auth_token = auth_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.pipeline = None
        
    def load_model(self):
        """Load the speaker diarization model"""
        if self.pipeline is not None:
            return
        
        console.print("[cyan]Loading Speaker Diarization model...[/cyan]")
        console.print("[yellow]⚠ This may take a few moments on first run...[/yellow]")
        
        try:
            # Load pre-trained speaker diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.auth_token
            )
            
            console.print("[green]✓ Diarization model loaded successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Error loading diarization model: {str(e)}[/red]")
            console.print("[yellow]Make sure you have accepted the model licenses at:[/yellow]")
            console.print("[yellow]1. https://huggingface.co/pyannote/speaker-diarization-3.1[/yellow]")
            console.print("[yellow]2. https://huggingface.co/pyannote/segmentation-3.0[/yellow]")
            console.print("[yellow]3. https://huggingface.co/pyannote/speaker-diarization-community-1[/yellow]")
            raise
    
    def diarize(self, audio_data, audio_path: str = None):
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_data: Audio data as dict {'waveform': tensor, 'sample_rate': int} or path
            audio_path: Path to audio file for logging
            
        Returns:
            Diarization result object
        """
        if self.pipeline is None:
            self.load_model()
        
        log_path = audio_path if audio_path else (audio_data if isinstance(audio_data, str) else "preloaded audio")
        console.print(f"[cyan]Performing speaker diarization on: {log_path}[/cyan]")
        
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
                
            # Run diarization
            result = self.pipeline(
                audio_data,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            
            # Handle new return type in pyannote.audio 3.1.0+ / 4.x
            # The pipeline might return a DiarizeOutput object instead of Annotation
            if hasattr(result, "speaker_diarization"):
                diarization = result.speaker_diarization
            else:
                diarization = result
            
            # Get number of speakers
            num_speakers = len(diarization.labels())
            
            console.print(f"[green]✓ Diarization complete: {num_speakers} speaker(s) detected[/green]")
            
            return diarization
            
        except Exception as e:
            console.print(f"[red]✗ Error during diarization: {str(e)}[/red]")
            raise
    
    def get_speaker_segments(self, diarization) -> Dict[str, List[tuple]]:
        """
        Extract speaker segments organized by speaker
        
        Args:
            diarization: Diarization result object
            
        Returns:
            Dictionary mapping speaker labels to list of (start, end) tuples
        """
        speaker_segments = {}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            
            speaker_segments[speaker].append((turn.start, turn.end))
        
        return speaker_segments
    
    def get_timeline(self, diarization) -> List[Dict]:
        """
        Get complete timeline of all speaker segments
        
        Args:
            diarization: Diarization result object
            
        Returns:
            List of dictionaries with speaker timeline
        """
        timeline = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end,
                'duration': turn.end - turn.start,
                'start_formatted': format_timestamp(turn.start),
                'end_formatted': format_timestamp(turn.end)
            })
        
        return timeline
    
    def get_speaker_stats(self, diarization) -> Dict[str, Dict]:
        """
        Get statistics for each speaker
        
        Args:
            diarization: Diarization result object
            
        Returns:
            Dictionary with speaker statistics
        """
        stats = {}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in stats:
                stats[speaker] = {
                    'total_duration': 0.0,
                    'num_segments': 0,
                    'segments': []
                }
            
            duration = turn.end - turn.start
            stats[speaker]['total_duration'] += duration
            stats[speaker]['num_segments'] += 1
            stats[speaker]['segments'].append({
                'start': turn.start,
                'end': turn.end,
                'duration': duration
            })
        
        return stats
    
    def save_timeline(self, timeline: List[Dict], output_path: str):
        """
        Save timeline to JSON file
        
        Args:
            timeline: Timeline data
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(timeline, f, indent=2)
            
            console.print(f"[green]✓ Timeline saved to: {output_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Error saving timeline: {str(e)}[/red]")
            raise
    
    def print_summary(self, diarization):
        """
        Print a summary of diarization results
        
        Args:
            diarization: Diarization result object
        """
        stats = self.get_speaker_stats(diarization)
        
        console.print("\n[bold cyan]═══ Speaker Summary ═══[/bold cyan]")
        
        for speaker, data in stats.items():
            console.print(f"\n[bold yellow]{speaker}:[/bold yellow]")
            console.print(f"  Total speaking time: {data['total_duration']:.2f}s")
            console.print(f"  Number of segments: {data['num_segments']}")
            console.print(f"  Average segment length: {data['total_duration']/data['num_segments']:.2f}s")
