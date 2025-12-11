"""
Medical NLP Pipeline - Clean version
"""
from typing import Dict, Any
from datetime import datetime
import json
import os

from .data_processor import TranscriptProcessor
from .ner_extractor import MedicalNERExtractor
from .summarizer import MedicalSummarizer
from .sentiment_analyzer import MedicalSentimentAnalyzer
from .soap_generator import SOAPGenerator

class MedicalNLPipeline:
    def __init__(self):
        print("=" * 60)
        print("Initializing Medical NLP Pipeline")
        print("=" * 60)
        
        # Initialize components
        self.transcript_processor = TranscriptProcessor()
        self.ner_extractor = MedicalNERExtractor()
        self.summarizer = MedicalSummarizer()
        self.sentiment_analyzer = MedicalSentimentAnalyzer()
        self.soap_generator = SOAPGenerator()
        
        print("✓ All components initialized successfully")
    
    def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """Process a medical transcript"""
        print("\n" + "=" * 60)
        print("Processing Medical Transcript")
        print("=" * 60)
        
        try:
            # Step 1: Parse transcript
            print("\n1. Parsing transcript...")
            dialogue_turns = self.transcript_processor.parse_transcript(transcript)
            patient_text = self.transcript_processor.extract_patient_dialogue(dialogue_turns)
            print(f"   ✓ Parsed {len(dialogue_turns)} dialogue turns")
            
            # Step 2: Extract entities
            print("\n2. Extracting medical entities...")
            entities = self.ner_extractor.extract_entities(transcript)
            keywords = self.ner_extractor.extract_keywords(transcript)
            print(f"   ✓ Extracted {sum(len(v) for v in entities.values())} medical entities")
            
            # Step 3: Generate summary
            print("\n3. Generating medical summary...")
            medical_summary = self.summarizer.generate_structured_summary(transcript, entities)
            print(f"   ✓ Generated structured summary")
            
            # Step 4: Analyze sentiment
            print("\n4. Analyzing patient sentiment...")
            sentiment_analysis = self.sentiment_analyzer.analyze_patient_sentiment(patient_text)
            print(f"   ✓ Sentiment: {sentiment_analysis.get('sentiment', 'UNKNOWN')}")
            
            # Step 5: Generate SOAP note
            print("\n5. Generating SOAP note...")
            soap_note = self.soap_generator.generate_soap_note(transcript, entities)
            print(f"   ✓ SOAP note generated")
            
            # Compile results
            results = {
                "timestamp": datetime.now().isoformat(),
                "processing_success": True,
                "medical_summary": medical_summary,
                "sentiment_analysis": sentiment_analysis,
                "soap_note": soap_note,
                "extracted_entities": entities,
                "keywords": keywords,
                "patient_text": patient_text,
                "dialogue_turns": [
                    {"speaker": turn.speaker, "text": turn.text}
                    for turn in dialogue_turns
                ]
            }
            
            print("\n" + "=" * 60)
            print("PROCESSING COMPLETE!")
            print("=" * 60)
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "processing_success": False,
                "error": str(e)
            }
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "output"):
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        full_path = os.path.join(output_dir, f"results_{timestamp}.json")
        with open(full_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to: {full_path}")
        return full_path