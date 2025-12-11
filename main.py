import json
from typing import Dict
from src.data_processor import TranscriptProcessor
from src.ner_extractor import MedicalNERExtractor
from src.summarizer import MedicalSummarizer
from src.sentiment_analyzer import MedicalSentimentAnalyzer
from src.soap_generator import SOAPGenerator
from src.utils import save_json, load_config

class MedicalNLPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        
        # Initialize components
        self.processor = TranscriptProcessor()
        self.ner_extractor = MedicalNERExtractor(
            model_name=self.config['models']['ner_model']
        )
        self.summarizer = MedicalSummarizer(
            model_name=self.config['models']['summarization_model']
        )
        self.sentiment_analyzer = MedicalSentimentAnalyzer(
            model_name=self.config['models']['sentiment_model']
        )
        self.soap_generator = SOAPGenerator()
    
    def process_transcript(self, transcript: str) -> Dict:
        """Main pipeline to process medical transcript"""
        
        # Step 1: Parse transcript
        dialogue_turns = self.processor.parse_transcript(transcript)
        patient_text = self.processor.extract_patient_dialogue(dialogue_turns)
        
        # Step 2: Extract medical entities
        entities = self.ner_extractor.extract_entities(transcript)
        keywords = self.ner_extractor.extract_keywords(transcript)
        
        # Step 3: Generate medical summary
        summary = self.summarizer.generate_structured_summary(transcript, entities)
        
        # Step 4: Analyze patient sentiment
        sentiment_analysis = self.sentiment_analyzer.analyze_patient_sentiment(patient_text)
        
        # Step 5: Generate SOAP note
        soap_note = self.soap_generator.generate_soap_note(transcript, entities)
        
        # Compile all results
        results = {
            "medical_summary": summary,
            "sentiment_analysis": sentiment_analysis,
            "soap_note": soap_note.to_dict(),
            "extracted_entities": entities,
            "keywords": keywords,
            "patient_text": patient_text,
            "dialogue_turns": [
                {"speaker": turn.speaker, "text": turn.text}
                for turn in dialogue_turns
            ]
        }
        
        return results
    
    def save_results(self, results: Dict, output_dir: str = "output"):
        """Save pipeline results to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results
        save_json(results, f"{output_dir}/full_results.json")
        
        # Save individual components
        save_json(results["medical_summary"], f"{output_dir}/medical_summary.json")
        save_json(results["sentiment_analysis"], f"{output_dir}/sentiment_analysis.json")
        save_json(results["soap_note"], f"{output_dir}/soap_note.json")
        save_json(results["extracted_entities"], f"{output_dir}/extracted_entities.json")
        
        print(f"Results saved to {output_dir}/ directory")

def main():
    """Main execution function"""
    
    # Sample transcript (from the prompt)
    transcript = """
    **Physician:** *Good morning, Ms. Jones. How are you feeling today?*
    
    **Patient:** *Good morning, doctor. I’m doing better, but I still have some discomfort now and then.*
    
    **Physician:** *I understand you were in a car accident last September. Can you walk me through what happened?*
    
    **Patient:** *Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.*
    
    **Physician:** *That sounds like a strong impact. Were you wearing your seatbelt?*
    
    **Patient:** *Yes, I always do.*
    
    **Physician:** *What did you feel immediately after the accident?*
    
    **Patient:** *At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.*
    
    **Physician:** *Did you seek medical attention at that time?*
    
    **Patient:** *Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home.*
    
    **Physician:** *How did things progress after that?*
    
    **Patient:** *The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.*
    
    **Physician:** *That makes sense. Are you still experiencing pain now?*
    
    **Patient:** *It’s not constant, but I do get occasional backaches. It’s nothing like before, though.*
    
    **Physician:** *That’s good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?*
    
    **Patient:** *No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.*
    
    **Physician:** *And how has this impacted your daily life? Work, hobbies, anything like that?*
    
    **Patient:** *I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.*
    
    **Physician:** *That’s encouraging. Let’s go ahead and do a physical examination to check your mobility and any lingering pain.*
    
    [**Physical Examination Conducted**]
    
    **Physician:** *Everything looks good. Your neck and back have a full range of movement, and there’s no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.*
    
    **Patient:** *That’s a relief!*
    
    **Physician:** *Yes, your recovery so far has been quite positive. Given your progress, I’d expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.*
    
    **Patient:** *That’s great to hear. So, I don’t need to worry about this affecting me in the future?*
    
    **Physician:** *That’s right. I don’t foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you’re on track for a full recovery.*
    
    **Patient:** *Thank you, doctor. I appreciate it.*
    
    **Physician:** *You’re very welcome, Ms. Jones. Take care, and don’t hesitate to reach out if you need anything.*
    """
    
    # Initialize pipeline
    pipeline = MedicalNLPipeline()
    
    # Process transcript
    print("Processing medical transcript...")
    results = pipeline.process_transcript(transcript)
    
    # Save results
    pipeline.save_results(results)
    
    # Display key results
    print("\n" + "="*50)
    print("MEDICAL SUMMARY:")
    print("="*50)
    print(json.dumps(results["medical_summary"], indent=2))
    
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS:")
    print("="*50)
    sentiment = results["sentiment_analysis"]
    print(f"Sentiment: {sentiment['sentiment']}")
    print(f"Intent: {sentiment['intent']}")
    print(f"Confidence: {sentiment['confidence']:.2f}")
    
    print("\n" + "="*50)
    print("SOAP NOTE:")
    print("="*50)
    print(json.dumps(results["soap_note"], indent=2))
    
    print("\n" + "="*50)
    print("EXTRACTED KEYWORDS:")
    print("="*50)
    print(results["keywords"][:10])

if __name__ == "__main__":
    main()