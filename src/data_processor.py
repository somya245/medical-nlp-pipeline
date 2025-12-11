"""
Transcript Processor - Clean version
"""
import re
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class DialogueTurn:
    speaker: str
    text: str

class TranscriptProcessor:
    def __init__(self):
        pass
    
    def parse_transcript(self, transcript: str) -> List[DialogueTurn]:
        """Parse transcript into structured dialogue turns"""
        dialogue_turns = []
        lines = transcript.strip().split('\n')
        
        current_speaker = None
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Match speaker pattern
            match = re.match(r'^(\*\*)?(?P<speaker>[A-Za-z\s]+)(\*\*)?:\s*\*?(?P<text>.*?)\*?$', line)
            
            if match:
                if current_speaker and current_text:
                    dialogue_turns.append(DialogueTurn(
                        speaker=current_speaker,
                        text=' '.join(current_text)
                    ))
                
                current_speaker = match.group('speaker').strip()
                current_text = [match.group('text').strip()]
            elif current_speaker:
                current_text.append(line)
        
        # Add the last turn
        if current_speaker and current_text:
            dialogue_turns.append(DialogueTurn(
                speaker=current_speaker,
                text=' '.join(current_text)
            ))
        
        return dialogue_turns
    
    def extract_patient_dialogue(self, dialogue_turns: List[DialogueTurn]) -> str:
        """Extract all patient dialogue"""
        patient_texts = []
        for turn in dialogue_turns:
            speaker_lower = turn.speaker.lower()
            if any(keyword in speaker_lower for keyword in ['patient', 'ms.', 'mr.', 'mrs.']):
                patient_texts.append(turn.text)
        return ' '.join(patient_texts)
    
    def extract_physician_dialogue(self, dialogue_turns: List[DialogueTurn]) -> str:
        """Extract all physician dialogue"""
        physician_texts = []
        for turn in dialogue_turns:
            speaker_lower = turn.speaker.lower()
            if any(keyword in speaker_lower for keyword in ['physician', 'doctor', 'dr.']):
                physician_texts.append(turn.text)
        return ' '.join(physician_texts)