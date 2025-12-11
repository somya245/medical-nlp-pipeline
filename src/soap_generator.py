"""
SOAP Note Generator - Clean version
"""
from typing import Dict, List
import re

class SOAPGenerator:
    def __init__(self):
        pass
    
    def generate_soap_note(self, transcript: str, entities: Dict) -> Dict:
        """Generate SOAP note from transcript"""
        
        # Extract subjective information
        subjective = self._extract_subjective(transcript)
        
        # Extract objective information
        objective = self._extract_objective(transcript)
        
        # Generate assessment
        assessment = self._generate_assessment(entities, subjective)
        
        # Generate plan
        plan = self._generate_plan(entities, assessment)
        
        return {
            "Subjective": subjective,
            "Objective": objective,
            "Assessment": assessment,
            "Plan": plan
        }
    
    def _extract_subjective(self, transcript: str) -> Dict:
        """Extract subjective information"""
        # Find patient statements
        patient_pattern = r'(?:Patient|Ms\.|Mr\.|Mrs\.):\s*(.*?)(?=\n\w+:|$)'
        patient_statements = re.findall(patient_pattern, transcript, re.DOTALL | re.IGNORECASE)
        
        # Extract chief complaint
        chief_complaint = self._extract_chief_complaint(' '.join(patient_statements))
        
        # Extract history
        history = self._extract_history(patient_statements)
        
        return {
            "Chief_Complaint": chief_complaint,
            "History_of_Present_Illness": history,
            "Patient_Reported_Symptoms": self._extract_reported_symptoms(patient_statements)
        }
    
    def _extract_objective(self, transcript: str) -> Dict:
        """Extract objective information"""
        # Look for examination findings
        exam_pattern = r'\[Physical Examination Conducted\](.*?)(?=\n\*\*|\Z)'
        exam_match = re.search(exam_pattern, transcript, re.DOTALL)
        
        if exam_match:
            exam_text = exam_match.group(1)
            return {
                "Physical_Exam": self._extract_exam_findings(exam_text),
                "Observations": "Patient appears in normal health",
                "Vital_Signs": "Not recorded"
            }
        
        # Fallback
        return {
            "Physical_Exam": "No detailed exam recorded in transcript",
            "Observations": "Based on conversation analysis",
            "Vital_Signs": "Not recorded"
        }
    
    def _generate_assessment(self, entities: Dict, subjective: Dict) -> Dict:
        """Generate assessment"""
        diagnosis = entities.get("DIAGNOSES", ["Unknown"])[0] if entities.get("DIAGNOSES") else "Unknown"
        
        # Determine severity
        symptoms = subjective.get("Patient_Reported_Symptoms", [])
        severity = "Mild"
        symptom_text = ' '.join(symptoms).lower()
        
        if any(severe in symptom_text for severe in ['severe', 'intense', 'unbearable', 'terrible']):
            severity = "Severe"
        elif any(moderate in symptom_text for moderate in ['moderate', 'significant', 'considerable']):
            severity = "Moderate"
        
        # Determine prognosis
        prognosis = "Good"
        if entities.get("PROGNOSES"):
            prognosis = entities["PROGNOSES"][0]
        
        return {
            "Diagnosis": diagnosis,
            "Severity": severity,
            "Prognosis": prognosis,
            "Differential_Diagnosis": ["Musculoskeletal strain", "Soft tissue injury", "Post-traumatic condition"]
        }
    
    def _generate_plan(self, entities: Dict, assessment: Dict) -> Dict:
        """Generate treatment plan"""
        treatments = entities.get("TREATMENTS", [])
        
        if not treatments:
            treatments = ["Conservative management", "Physical therapy as needed"]
        
        return {
            "Treatment": treatments,
            "Follow_Up": "Return if symptoms worsen or persist beyond recommended timeframe",
            "Medications": ["Analgesics for pain relief as needed"],
            "Referrals": ["Physical therapy"] if "physiotherapy" not in ' '.join(treatments).lower() else []
        }
    
    def _extract_chief_complaint(self, text: str) -> str:
        """Extract chief complaint"""
        complaint_keywords = ['pain', 'discomfort', 'hurt', 'ache', 'injury', 'problem', 'issue']
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in complaint_keywords):
                return sentence.strip()[:100]
        
        return "Medical complaint (specifics not detailed)"
    
    def _extract_history(self, statements: List[str]) -> str:
        """Extract history"""
        timeline_phrases = ['accident', 'happened', 'occurred', 'since', 'after', 'weeks', 'months', 'ago']
        
        relevant_statements = []
        for statement in statements:
            if any(phrase in statement.lower() for phrase in timeline_phrases):
                clean_stmt = statement.replace('*', '').strip()
                if clean_stmt and clean_stmt not in relevant_statements:
                    relevant_statements.append(clean_stmt)
        
        if relevant_statements:
            return ' '.join(relevant_statements[:3])
        
        return "Patient history details not fully described"
    
    def _extract_reported_symptoms(self, statements: List[str]) -> List[str]:
        """Extract reported symptoms"""
        symptoms = []
        symptom_keywords = ['pain', 'discomfort', 'stiffness', 'ache', 'sore', 'tender']
        
        for statement in statements:
            words = statement.lower().split()
            for i, word in enumerate(words):
                if word in symptom_keywords:
                    # Get context
                    start = max(0, i-2)
                    end = min(len(words), i+3)
                    symptom_phrase = ' '.join(words[start:end])
                    symptoms.append(symptom_phrase)
        
        return list(set(symptoms))[:5]
    
    def _extract_exam_findings(self, exam_text: str) -> str:
        """Extract exam findings"""
        findings = []
        exam_keywords = ['mobility', 'movement', 'tenderness', 'range', 'condition', 'normal', 'good']
        
        sentences = re.split(r'[.!?]+', exam_text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in exam_keywords):
                findings.append(sentence.strip())
        
        if findings:
            return ' '.join(findings[:3])
        
        return "Physical examination findings not detailed"