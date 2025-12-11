"""
Medical Sentiment Analyzer - Clean version
"""
from typing import Dict
import re

class MedicalSentimentAnalyzer:
    def __init__(self):
        # Medical sentiment categories
        self.sentiment_keywords = {
            "ANXIOUS": ['worried', 'anxious', 'nervous', 'scared', 'afraid', 
                       'concerned', 'stressed', 'tense', 'apprehensive'],
            "NEUTRAL": ['okay', 'fine', 'normal', 'alright', 'stable'],
            "REASSURED": ['better', 'improved', 'relief', 'happy', 'great',
                         'relieved', 'good', 'pleased', 'satisfied'],
            "PAINFUL": ['pain', 'hurt', 'ache', 'discomfort', 'sore',
                       'tender', 'throbbing', 'sharp', 'dull']
        }
        
        # Intent keywords
        self.intent_keywords = {
            "seeking_reassurance": ['worry', 'concern', 'anxious', 'nervous', 
                                   'hope', 'apprehensive', 'uncertain'],
            "reporting_symptoms": ['pain', 'hurt', 'discomfort', 'symptom', 
                                  'feel', 'experience', 'notice'],
            "expressing_gratitude": ['thank', 'appreciate', 'grateful', 
                                    'relief', 'happy', 'thanks'],
            "expressing_concern": ['afraid', 'scared', 'terrified', 
                                  'panic', 'dread', 'frightened']
        }
    
    def analyze_patient_sentiment(self, patient_text: str) -> Dict:
        """Analyze patient sentiment and intent"""
        text_lower = patient_text.lower()
        
        # Calculate sentiment scores
        sentiment_scores = {sentiment: 0 for sentiment in self.sentiment_keywords}
        
        for sentiment, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    sentiment_scores[sentiment] += 1
        
        # Determine dominant sentiment
        total_score = sum(sentiment_scores.values())
        if total_score > 0:
            dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[dominant_sentiment] / total_score
        else:
            dominant_sentiment = "NEUTRAL"
            confidence = 0.5
        
        # Detect intent
        intent_scores = {intent: 0 for intent in self.intent_keywords}
        
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    intent_scores[intent] += 1
        
        if sum(intent_scores.values()) > 0:
            dominant_intent = max(intent_scores, key=intent_scores.get)
        else:
            dominant_intent = "general_conversation"
        
        # Map intent to readable format
        intent_map = {
            "seeking_reassurance": "Seeking reassurance",
            "reporting_symptoms": "Reporting symptoms",
            "expressing_gratitude": "Expressing gratitude",
            "expressing_concern": "Expressing concern",
            "general_conversation": "General conversation"
        }
        
        return {
            "sentiment": dominant_sentiment,
            "sentiment_scores": sentiment_scores,
            "intent": intent_map.get(dominant_intent, "General conversation"),
            "confidence": round(confidence, 2)
        }