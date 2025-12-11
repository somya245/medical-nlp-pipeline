import json
from typing import Dict, Any
import yaml

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_json(data: Dict, filepath: str) -> None:
    """Save data as JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Dict:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def format_output(data: Dict, format_type: str = "json") -> str:
    """Format output in specified format"""
    if format_type == "json":
        return json.dumps(data, indent=2)
    elif format_type == "yaml":
        return yaml.dump(data, default_flow_style=False)
    else:
        return str(data)

def validate_transcript(transcript: str) -> bool:
    """Validate transcript format"""
    required_speakers = ['physician', 'patient']
    transcript_lower = transcript.lower()
    
    has_physician = any(speaker in transcript_lower for speaker in ['physician', 'doctor'])
    has_patient = any(speaker in transcript_lower for speaker in ['patient', 'ms.', 'mr.'])
    
    return has_physician and has_patient