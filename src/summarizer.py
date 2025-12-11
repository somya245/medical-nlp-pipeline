"""
Medical Summarizer - Clean version
"""
from typing import Dict, List
import re

class MedicalSummarizer:
    def __init__(self):
        pass
    
    def generate_structured_summary(self, transcript: str, entities: Dict) -> Dict:
        """Generate structured medical summary"""
        
        # Extract patient name
        patient_name = "Unknown"
        name_patterns = [
            r'(?:Ms\.|Mr\.|Mrs\.)\s+([A-Z][a-z]+)',
            r'Patient:\s*([A-Z][a-z]+)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                patient_name = match.group(1)
                break
        
        # Generate text summary
        text_summary = self._generate_text_summary(transcript)
        
        # Build structured summary
        summary = {
            "Patient_Name": patient_name,
            "Symptoms": entities.get("SYMPTOMS", []),
            "Diagnosis": entities.get("DIAGNOSES", ["Unknown"])[0] if entities.get("DIAGNOSES") else "Unknown",
            "Treatment": entities.get("TREATMENTS", []),
            "Current_Status": self._extract_current_status(text_summary),
            "Prognosis": entities.get("PROGNOSES", []),
            "Timeline": entities.get("DATES", []),
            "Summary": text_summary
        }
        
        return summary
    
    def _generate_text_summary(self, text: str) -> str:
        """Generate text summary"""
        # Simple extractive summarization
        sentences = re.split(r'[.!?]+', text)
        
        # Score sentences based on medical keywords
        medical_keywords = ['pain', 'accident', 'injury', 'treatment', 'therapy',
                           'recovery', 'whiplash', 'fracture', 'hospital', 'doctor']
        
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            for keyword in medical_keywords:
                if keyword.lower() in sentence.lower():
                    score += 1
            if score > 0:
                scored_sentences.append((score, i, sentence.strip()))
        
        # Sort by score and take top 3
        scored_sentences.sort(reverse=True)
        top_sentences = [sentence for _, _, sentence in scored_sentences[:3]]
        
        return ' '.join(top_sentences) if top_sentences else "Summary not available"
    
    def _extract_current_status(self, summary: str) -> str:
        """Extract current status from summary"""
        status_keywords = {
            'improving': 'Condition improving',
            'better': 'Feeling better',
            'recovered': 'Fully recovered',
            'stable': 'Condition stable',
            'worsening': 'Condition worsening',
            'pain': 'Still experiencing pain'
        }
        
        summary_lower = summary.lower()
        for keyword, status in status_keywords.items():
            if keyword in summary_lower:
                return status
        
        return "Condition status not specified"