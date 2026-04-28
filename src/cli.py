"""
Command-Line Interface for Human Voice Detection System
"""
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys

from src.audio_processor import AudioProcessor
from src.voice_detector import VoiceDetector
from src.speaker_diarization import SpeakerDiarizer
from src.audio_separator import AudioSeparator
from src.utils import validate_audio_file, ensure_dir, format_timestamp
from config import Config

console = Console()


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Human Voice Detection and Recognition System
    
    Detect human voices and separate different speakers from audio/video files.
    """
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='output', help='Output directory')
@click.option('--min-speakers', default=1, help='Minimum number of speakers')
@click.option('--max-speakers', default=10, help='Maximum number of speakers')
@click.option('--detect-only', is_flag=True, help='Only detect voice, skip diarization')
def process(input_file, output, min_speakers, max_speakers, detect_only):
    """Process an audio or video file to detect and separate speakers"""
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        console.print(f"[red]✗ Configuration Error: {str(e)}[/red]")
        console.print("\n[yellow]Setup Instructions:[/yellow]")
        console.print("1. Copy .env.example to .env")
        console.print("2. Get your HuggingFace token from: https://huggingface.co/settings/tokens")
        console.print("3. Accept model license at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        console.print("4. Add your token to .env file")
        sys.exit(1)
    
    # Validate input file
    is_valid, message = validate_audio_file(input_file)
    if not is_valid:
        console.print(f"[red]✗ {message}[/red]")
        sys.exit(1)
    
    # Create output directory
    output_dir = ensure_dir(output)
    
    console.print("\n[bold cyan]═══ Human Voice Detection & Recognition ═══[/bold cyan]\n")
    console.print(f"[cyan]Input file:[/cyan] {input_file}")
    console.print(f"[cyan]Output directory:[/cyan] {output_dir}\n")
    
    try:
        # Initialize processors
        audio_processor = AudioProcessor(sample_rate=Config.SAMPLE_RATE)
        voice_detector = VoiceDetector(auth_token=Config.HUGGINGFACE_TOKEN)
        
        # Process audio/video file
        audio_path, audio_data, sample_rate = audio_processor.process_file(input_file)
        
        # Detect voice activity
        vad_result = voice_detector.detect_voice({'waveform': audio_data, 'sample_rate': sample_rate}, audio_path)
        voice_segments = voice_detector.get_voice_segments(vad_result)
        
        console.print(f"\n[green]✓ Voice Activity Detected:[/green] {len(voice_segments)} segments")
        
        if detect_only:
            # Save voice segments info
            import json
            segments_data = [
                {
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'start_formatted': format_timestamp(start),
                    'end_formatted': format_timestamp(end)
                }
                for start, end in voice_segments
            ]
            
            output_file = output_dir / 'voice_segments.json'
            with open(output_file, 'w') as f:
                json.dump(segments_data, f, indent=2)
            
            console.print(f"[green]✓ Voice segments saved to: {output_file}[/green]")
            return
        
        # Perform speaker diarization
        console.print("\n[cyan]Starting speaker diarization...[/cyan]")
        diarizer = SpeakerDiarizer(
            auth_token=Config.HUGGINGFACE_TOKEN,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        diarization = diarizer.diarize({'waveform': audio_data, 'sample_rate': sample_rate}, audio_path)
        
        # Print summary
        diarizer.print_summary(diarization)
        
        # Get speaker information
        speaker_segments = diarizer.get_speaker_segments(diarization)
        timeline = diarizer.get_timeline(diarization)
        
        # Save timeline
        timeline_file = output_dir / 'timeline.json'
        diarizer.save_timeline(timeline, str(timeline_file))
        
        # Separate speakers
        console.print("\n[cyan]Separating speakers...[/cyan]")
        separator = AudioSeparator(sample_rate=Config.SAMPLE_RATE)
        
        input_name = Path(input_file).stem
        output_files = separator.separate_speakers(
            audio_data,
            speaker_segments,
            str(output_dir),
            base_name=f"{input_name}_speaker"
        )
        
        # Create summary table
        console.print("\n[bold cyan]═══ Results Summary ═══[/bold cyan]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Speaker", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Duration", justify="right", style="yellow")
        
        stats = diarizer.get_speaker_stats(diarization)
        for speaker, filepath in output_files.items():
            duration = stats[speaker]['total_duration']
            filename = Path(filepath).name
            table.add_row(speaker, filename, f"{duration:.2f}s")
        
        console.print(table)
        
        console.print(f"\n[bold green]✓ Processing complete![/bold green]")
        console.print(f"[green]Output saved to: {output_dir}[/green]\n")
        
    except Exception as e:
        console.print(f"\n[red]✗ Error: {str(e)}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
def check(input_file):
    """Quickly check if an audio/video file contains human voice"""
    
    try:
        Config.validate()
    except ValueError as e:
        console.print(f"[red]✗ Configuration Error: {str(e)}[/red]")
        sys.exit(1)
    
    is_valid, message = validate_audio_file(input_file)
    if not is_valid:
        console.print(f"[red]✗ {message}[/red]")
        sys.exit(1)
    
    console.print(f"\n[cyan]Checking:[/cyan] {input_file}\n")
    
    try:
        audio_processor = AudioProcessor(sample_rate=Config.SAMPLE_RATE)
        voice_detector = VoiceDetector(auth_token=Config.HUGGINGFACE_TOKEN)
        
        audio_path, audio_data, sample_rate = audio_processor.process_file(input_file)
        has_voice = voice_detector.has_voice({'waveform': audio_data, 'sample_rate': sample_rate}, audio_path)
        
        if has_voice:
            console.print("[bold green]✓ Human voice DETECTED[/bold green]\n")
        else:
            console.print("[bold yellow]✗ No human voice detected[/bold yellow]\n")
        
    except Exception as e:
        console.print(f"[red]✗ Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
def setup():
    """Display setup instructions"""
    
    console.print("\n[bold cyan]═══ Setup Instructions ═══[/bold cyan]\n")
    
    console.print("[yellow]1. Install Dependencies:[/yellow]")
    console.print("   pip install -r requirements.txt\n")
    
    console.print("[yellow]2. Get HuggingFace Token:[/yellow]")
    console.print("   - Visit: https://huggingface.co/settings/tokens")
    console.print("   - Create a new token (read access is sufficient)\n")
    
    console.print("[yellow]3. Accept Model Licenses:[/yellow]")
    console.print("   - Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
    console.print("   - Click 'Accept' on the license agreement")
    console.print("   - Visit: https://huggingface.co/pyannote/segmentation-3.0")
    console.print("   - Click 'Accept' on the license agreement")
    console.print("   - Visit: https://huggingface.co/pyannote/speaker-diarization-community-1")
    console.print("   - Click 'Accept' on the license agreement (required for PLDA)\n")
    
    console.print("[yellow]4. Configure Environment:[/yellow]")
    console.print("   - Copy .env.example to .env")
    console.print("   - Add your HuggingFace token to the .env file\n")
    
    console.print("[green]Then you're ready to go![/green]\n")


if __name__ == '__main__':
    cli()
