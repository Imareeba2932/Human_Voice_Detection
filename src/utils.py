"""
Utility functions for the Voice Detection System
"""
import os
from pathlib import Path
from typing import Tuple
from datetime import timedelta


def ensure_dir(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        directory: Directory path to ensure exists
        
    Returns:
        Path object of the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS.mmm format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = td.total_seconds() % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def get_file_extension(filepath: str) -> str:
    """
    Get file extension from filepath
    
    Args:
        filepath: Path to file
        
    Returns:
        File extension (lowercase, without dot)
    """
    return Path(filepath).suffix.lower().strip('.')


def validate_audio_file(filepath: str) -> Tuple[bool, str]:
    """
    Validate if file exists and is a supported audio/video format
    
    Args:
        filepath: Path to file to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    supported_audio = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac']
    supported_video = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm']
    
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    ext = get_file_extension(filepath)
    
    if ext in supported_audio:
        return True, "Valid audio file"
    elif ext in supported_video:
        return True, "Valid video file"
    else:
        return False, f"Unsupported file format: {ext}. Supported formats: {', '.join(supported_audio + supported_video)}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def get_output_path(input_file: str, output_dir: str, suffix: str = "", extension: str = "wav") -> str:
    """
    Generate output file path based on input file
    
    Args:
        input_file: Path to input file
        output_dir: Output directory
        suffix: Suffix to add to filename
        extension: Output file extension
        
    Returns:
        Output file path
    """
    input_path = Path(input_file)
    base_name = input_path.stem
    
    if suffix:
        output_name = f"{base_name}_{suffix}.{extension}"
    else:
        output_name = f"{base_name}.{extension}"
    
    output_name = sanitize_filename(output_name)
    return str(Path(output_dir) / output_name)
