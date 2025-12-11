"""
Medical NLP Pipeline - FIXED SINGLE FILE VERSION
No import errors, no image errors - everything works
Run with: streamlit run medical_nlp_app.py
"""

import streamlit as st
import re
import json
from datetime import datetime
import os
import sys
import base64
from typing import List, Dict, Any

# ========== ALL PIPELINE COMPONENTS IN ONE FILE ==========

# 1. Data Processor
class TranscriptProcessor:
    def parse_transcript(self, transcript: str) -> List[Dict]:
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
                    dialogue_turns.append({
                        'speaker': current_speaker,
                        'text': ' '.join(current_text)
                    })
                
                current_speaker = match.group('speaker').strip()
                current_text = [match.group('text').strip()]
            elif current_speaker:
                current_text.append(line)
        
        # Add the last turn
        if current_speaker and current_text:
            dialogue_turns.append({
                'speaker': current_speaker,
                'text': ' '.join(current_text)
            })
        
        return dialogue_turns
    
    def extract_patient_dialogue(self, dialogue_turns: List[Dict]) -> str:
        """Extract all patient dialogue"""
        patient_texts = []
        for turn in dialogue_turns:
            speaker_lower = turn['speaker'].lower()
            if any(keyword in speaker_lower for keyword in ['patient', 'ms.', 'mr.', 'mrs.']):
                patient_texts.append(turn['text'])
        return ' '.join(patient_texts)

# 2. Medical NER Extractor
class MedicalNERExtractor:
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using regex"""
        text_lower = text.lower()
        
        entities = {
            "SYMPTOMS": [],
            "DIAGNOSES": [],
            "TREATMENTS": [],
            "PROGNOSES": [],
            "BODY_PARTS": [],
            "DATES": [],
            "LOCATIONS": []
        }
        
        # Medical entity patterns
        patterns = {
            "SYMPTOMS": [
                r'\b(pain|ache|discomfort|stiffness|soreness|tenderness)\b',
                r'\b(headache|backache|neck pain|back pain)\b',
                r'\b(nausea|dizziness|fatigue|weakness|numbness|tingling)\b'
            ],
            "DIAGNOSES": [
                r'\b(whiplash injury|whiplash)\b',
                r'\b(muscle strain|ligament sprain)\b',
                r'\b(fracture|broken bone)\b',
                r'\b(concussion|head injury)\b'
            ],
            "TREATMENTS": [
                r'\b(physiotherapy|physical therapy)\b',
                r'\b(painkillers|medication)\b',
                r'\b(surgery|operation)\b'
            ],
            "BODY_PARTS": [
                r'\b(neck|back|head|spine|shoulder|knee)\b'
            ],
            "DATES": [
                r'\b(september|october|november|december)\b',
                r'\b(\d+\s+(weeks?|months?|years?) ago)\b'
            ],
            "LOCATIONS": [
                r'\b(hospital|clinic|ER|A&E)\b'
            ],
            "PROGNOSES": [
                r'\b(full recovery|complete recovery)\b',
                r'\b(improving|getting better)\b'
            ]
        }
        
        # Extract entities
        for category, category_patterns in patterns.items():
            for pattern in category_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    if match and match not in entities[category]:
                        entities[category].append(match.title())
        
        # Remove duplicates
        for category in entities:
            entities[category] = list(dict.fromkeys(entities[category]))
        
        return entities
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords"""
        keywords = []
        text_lower = text.lower()
        
        medical_words = [
            'pain', 'accident', 'injury', 'treatment', 'therapy',
            'recovery', 'whiplash', 'back', 'neck', 'head',
            'physiotherapy', 'medication', 'hospital', 'doctor'
        ]
        
        for word in medical_words:
            if word in text_lower and word not in keywords:
                keywords.append(word.title())
                if len(keywords) >= top_n:
                    break
        
        return keywords

# 3. Sentiment Analyzer
class MedicalSentimentAnalyzer:
    def analyze_patient_sentiment(self, patient_text: str) -> Dict[str, Any]:
        """Analyze patient sentiment"""
        text_lower = patient_text.lower()
        
        # Sentiment keywords
        sentiment_words = {
            "ANXIOUS": ['worried', 'anxious', 'nervous', 'scared', 'afraid'],
            "REASSURED": ['better', 'improved', 'relief', 'happy', 'great'],
            "PAINFUL": ['pain', 'hurt', 'ache', 'discomfort', 'sore'],
            "NEUTRAL": ['okay', 'fine', 'normal', 'alright']
        }
        
        # Count occurrences
        scores = {sentiment: 0 for sentiment in sentiment_words}
        
        for sentiment, words in sentiment_words.items():
            for word in words:
                if word in text_lower:
                    scores[sentiment] += 1
        
        # Determine sentiment
        total = sum(scores.values())
        if total > 0:
            dominant = max(scores, key=scores.get)
            confidence = scores[dominant] / total
        else:
            dominant = "NEUTRAL"
            confidence = 0.5
        
        # Determine intent
        intent = "General conversation"
        if any(word in text_lower for word in ['worried', 'concerned', 'anxious']):
            intent = "Seeking reassurance"
        elif any(word in text_lower for word in ['pain', 'hurt', 'symptom']):
            intent = "Reporting symptoms"
        elif any(word in text_lower for word in ['thank', 'appreciate']):
            intent = "Expressing gratitude"
        
        return {
            "sentiment": dominant,
            "intent": intent,
            "confidence": round(confidence, 2),
            "scores": scores
        }

# 4. SOAP Generator
class SOAPGenerator:
    def generate_soap_note(self, transcript: str, entities: Dict) -> Dict[str, Any]:
        """Generate SOAP note"""
        # Extract patient name
        patient_name = "Unknown"
        name_match = re.search(r'(?:Ms\.|Mr\.|Mrs\.)\s+([A-Z][a-z]+)', transcript, re.IGNORECASE)
        if name_match:
            patient_name = name_match.group(1)
        else:
            # Try to find name from context
            name_match = re.search(r'Patient:\s*([A-Z][a-z]+)', transcript, re.IGNORECASE)
            if name_match:
                patient_name = name_match.group(1)
        
        # Build SOAP note
        return {
            "Subjective": {
                "Chief_Complaint": self._extract_complaint(transcript),
                "History_of_Present_Illness": self._extract_history(transcript),
                "Patient_Name": patient_name
            },
            "Objective": {
                "Physical_Exam": "Based on patient description - detailed physical exam not recorded",
                "Observations": "Patient appears coherent and responsive based on dialogue"
            },
            "Assessment": {
                "Diagnosis": entities.get("DIAGNOSES", ["Condition requiring evaluation"])[0] if entities.get("DIAGNOSES") else "Condition requiring evaluation",
                "Severity": self._determine_severity(transcript),
                "Prognosis": entities.get("PROGNOSES", ["Favorable with appropriate care"])[0] if entities.get("PROGNOSES") else "Favorable with appropriate care"
            },
            "Plan": {
                "Treatment": entities.get("TREATMENTS", ["Conservative management recommended"]) or ["Conservative management recommended"],
                "Follow_Up": "Follow up as needed based on symptom progression",
                "Medications": ["Analgesics as needed for pain management"]
            }
        }
    
    def _extract_complaint(self, text: str) -> str:
        """Extract chief complaint"""
        complaints = []
        complaint_keywords = ['pain', 'discomfort', 'hurt', 'ache', 'injury', 'problem']
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in complaint_keywords):
                # Clean and shorten
                clean_sentence = re.sub(r'\*+', '', sentence).strip()
                if len(clean_sentence) > 100:
                    clean_sentence = clean_sentence[:100] + "..."
                complaints.append(clean_sentence)
        
        return complaints[0] if complaints else "Patient reports medical issue"
    
    def _extract_history(self, text: str) -> str:
        """Extract history"""
        sentences = re.split(r'[.!?]+', text)
        relevant = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in ['accident', 'happened', 'occurred', 'since', 'after', 'weeks', 'months', 'ago', 'injured']):
                # Clean the sentence
                clean_sentence = re.sub(r'\*+', '', sentence).strip()
                if clean_sentence and clean_sentence not in relevant:
                    relevant.append(clean_sentence)
        
        if relevant:
            # Join and limit length
            history = ' '.join(relevant[:3])
            if len(history) > 200:
                history = history[:200] + "..."
            return history
        
        return "Patient history details provided in conversation"
    
    def _determine_severity(self, text: str) -> str:
        """Determine condition severity"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['severe', 'terrible', 'unbearable', 'excruciating']):
            return "Severe"
        elif any(word in text_lower for word in ['moderate', 'significant', 'considerable']):
            return "Moderate"
        elif any(word in text_lower for word in ['mild', 'slight', 'minor']):
            return "Mild"
        
        return "Moderate"  # Default

# 5. Summarizer
class MedicalSummarizer:
    def generate_structured_summary(self, transcript: str, entities: Dict) -> Dict[str, Any]:
        """Generate medical summary"""
        # Extract patient name
        patient_name = "Unknown"
        name_match = re.search(r'(?:Ms\.|Mr\.|Mrs\.)\s+([A-Z][a-z]+)', transcript, re.IGNORECASE)
        if name_match:
            patient_name = name_match.group(1)
        
        # Generate summary text
        summary_text = self._generate_summary_text(transcript)
        
        return {
            "Patient_Name": patient_name,
            "Symptoms": entities.get("SYMPTOMS", []),
            "Diagnosis": entities.get("DIAGNOSES", ["Requires medical evaluation"])[0] if entities.get("DIAGNOSES") else "Requires medical evaluation",
            "Treatment": entities.get("TREATMENTS", []),
            "Current_Status": self._extract_status(summary_text),
            "Prognosis": entities.get("PROGNOSES", ["Good with proper care"]),
            "Timeline": entities.get("DATES", []),
            "Summary": summary_text
        }
    
    def _generate_summary_text(self, text: str) -> str:
        """Generate summary text"""
        sentences = re.split(r'[.!?]+', text)
        
        # Score sentences based on medical relevance
        medical_keywords = ['pain', 'accident', 'injury', 'treatment', 'whiplash', 
                          'recovery', 'symptom', 'diagnosis', 'therapy', 'hospital']
        scored = []
        
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            for keyword in medical_keywords:
                if keyword in sentence_lower:
                    score += 1
            
            if score > 0:
                # Clean the sentence
                clean_sentence = re.sub(r'\*+', '', sentence).strip()
                if clean_sentence:
                    scored.append((score, i, clean_sentence))
        
        # Get top sentences
        scored.sort(reverse=True)
        top_sentences = [sentence for _, _, sentence in scored[:3]]
        
        if top_sentences:
            summary = ' '.join(top_sentences)
            if len(summary) > 300:
                summary = summary[:300] + "..."
            return summary
        
        return "Medical conversation analyzed. Patient discussing health concerns."
    
    def _extract_status(self, summary: str) -> str:
        """Extract status from summary"""
        summary_lower = summary.lower()
        
        if any(word in summary_lower for word in ['better', 'improved', 'improving', 'recovering']):
            return "Condition improving"
        elif any(word in summary_lower for word in ['pain', 'hurt', 'ache', 'discomfort']):
            return "Experiencing symptoms"
        elif any(word in summary_lower for word in ['recovered', 'healed', 'resolved']):
            return "Condition resolved"
        
        return "Under evaluation"

# 6. Main Pipeline
class MedicalNLPipeline:
    def __init__(self):
        self.transcript_processor = TranscriptProcessor()
        self.ner_extractor = MedicalNERExtractor()
        self.sentiment_analyzer = MedicalSentimentAnalyzer()
        self.soap_generator = SOAPGenerator()
        self.summarizer = MedicalSummarizer()
        print("Medical NLP Pipeline initialized successfully")
    
    def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """Process transcript through pipeline"""
        try:
            # Step 1: Parse transcript
            dialogue_turns = self.transcript_processor.parse_transcript(transcript)
            patient_text = self.transcript_processor.extract_patient_dialogue(dialogue_turns)
            
            # Step 2: Extract entities
            entities = self.ner_extractor.extract_entities(transcript)
            keywords = self.ner_extractor.extract_keywords(transcript)
            
            # Step 3: Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_patient_sentiment(patient_text)
            
            # Step 4: Generate SOAP note
            soap_note = self.soap_generator.generate_soap_note(transcript, entities)
            
            # Step 5: Generate summary
            summary = self.summarizer.generate_structured_summary(transcript, entities)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "processing_success": True,
                "medical_summary": summary,
                "sentiment_analysis": sentiment,
                "soap_note": soap_note,
                "extracted_entities": entities,
                "keywords": keywords,
                "patient_text": patient_text,
                "dialogue_turns": dialogue_turns
            }
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "processing_success": False,
                "error": str(e)
            }
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "output") -> str:
        """Save results to file"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"medical_results_{timestamp}.json")
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            return filename
        except Exception as e:
            return f"Error saving file: {str(e)}"

# ========== STREAMLIT APP ==========

# Page configuration
st.set_page_config(
    page_title="Medical NLP Pipeline",
    page_icon="📋",  # Using text icon instead of emoji file
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
# Custom CSS for styling
# Custom CSS for styling - UPDATED VERSION
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .symptom-badge {
        background-color: #DBEAFE;
        color: #1E40AF !important;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid #BFDBFE;
    }
    /* WHITE BOX STYLING */
    .white-box {
        background-color: white;
        border: 1px solid #E5E7EB;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .white-box-title {
        font-size: 1.1rem;
        color: #1E40AF;
        margin-bottom: 0.8rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    /* Fix for text visibility */
    .diagnosis-text {
        color: #92400E !important;
        font-weight: 600;
    }
    /* Ensure all text in boxes is visible */
    div[data-testid="stMarkdownContainer"] p {
        color: #1F2937 !important;
    }
    /* WHITE BOX OVERRIDES */
    .element-container .stMarkdown p {
        color: #1F2937 !important;
    }
    /* Section headers with icons */
    .section-with-icon {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
        color: #1E40AF;
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = MedicalNLPipeline()

if 'results' not in st.session_state:
    st.session_state.results = None

if 'transcript' not in st.session_state:
    st.session_state.transcript = ""

# Title Section
st.markdown('<h1 class="main-title">🏥 Medical NLP Pipeline</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #4B5563;">
        AI-powered analysis of medical conversations for symptoms, sentiment, and clinical documentation
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 style="color: #1E40AF;">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    # Input Options with icon - REPLACE the simple title line
    st.markdown('<div class="section-with-icon"><span>📥</span><span>Input Options</span></div>', unsafe_allow_html=True)
    
    input_method = st.radio(
        "Choose input method:",
        ["📝 Sample Transcript", "📋 Paste Your Text", "📁 Upload File"],
        key="input_method"
    )
    
    # Handle different input methods
    if input_method == "📝 Sample Transcript":
        sample_option = st.selectbox(
            "Select a sample case:",
            ["Car Accident & Whiplash", "Chronic Back Pain", "Post-Surgery Follow-up", "General Checkup"],
            key="sample_option"
        )
        
        if sample_option == "Car Accident & Whiplash":
            st.session_state.transcript = """**Physician:** Good morning, Ms. Jones. How are you feeling today?

**Patient:** Good morning, doctor. I'm doing better, but I still have some discomfort now and then.

**Physician:** I understand you were in a car accident last September. Can you walk me through what happened?

**Patient:** Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.

**Physician:** That sounds like a strong impact. Were you wearing your seatbelt?

**Patient:** Yes, I always do.

**Physician:** What did you feel immediately after the accident?

**Patient:** At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.

**Physician:** Did you seek medical attention at that time?

**Patient:** Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn't do any X-rays. They just gave me some advice and sent me home.

**Physician:** How did things progress after that?

**Patient:** The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.

**Physician:** That makes sense. Are you still experiencing pain now?

**Patient:** It's not constant, but I do get occasional backaches. It's nothing like before, though.

**Physician:** That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?

**Patient:** No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.

**Physician:** And how has this impacted your daily life? Work, hobbies, anything like that?

**Patient:** I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from doing anything.

**Physician:** That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.

[**Physical Examination Conducted**]

**Physician:** Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.

**Patient:** That's a relief!

**Physician:** Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.

**Patient:** That's great to hear. So, I don't need to worry about this affecting me in the future?

**Physician:** That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.

**Patient:** Thank you, doctor. I appreciate it.

**Physician:** You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything."""
        
        elif sample_option == "Chronic Back Pain":
            st.session_state.transcript = """**Physician:** Hello, I understand you're here about back pain. Can you tell me more?

**Patient:** Yes, I've been having lower back pain for about three months now. It started gradually but has been getting worse.

**Physician:** Can you describe the pain?

**Patient:** It's a dull ache that gets worse when I sit for long periods or bend over. Sometimes it radiates down my right leg.

**Physician:** Have you tried any treatments so far?

**Patient:** I've been taking ibuprofen, which helps a little. I also tried some stretching exercises from online videos.

**Physician:** How has this affected your daily activities?

**Patient:** I can't garden anymore, which I love, and I have trouble sleeping through the night."""
        
        elif sample_option == "Post-Surgery Follow-up":
            st.session_state.transcript = """**Physician:** Welcome back. How are you recovering from the knee surgery?

**Patient:** Much better than last week, thank you. The swelling has gone down quite a bit.

**Physician:** That's good to hear. How's the pain level?

**Patient:** It's manageable with the medication. About a 3 out of 10 most of the time.

**Physician:** Are you able to do the physical therapy exercises?

**Patient:** Yes, I'm doing them twice a day as instructed. They're challenging but getting easier."""
        
        else:  # General Checkup
            st.session_state.transcript = """**Physician:** Good to see you for your annual checkup. How have you been feeling overall?

**Patient:** Generally good, doctor. No major complaints.

**Physician:** Any changes in your health since we last met?

**Patient:** Just the usual seasonal allergies, but otherwise everything seems normal.

**Physician:** That's great. Let's go through the routine checks then."""
    
    elif input_method == "📋 Paste Your Text":
        st.session_state.transcript = st.text_area(
            "Paste your medical transcript below:",
            value=st.session_state.transcript,
            height=250,
            placeholder="Paste doctor-patient conversation here...\n\nExample format:\n**Physician:** Question here?\n**Patient:** Answer here.\n**Physician:** Follow-up question?\n**Patient:** Detailed response.",
            key="text_area"
        )
    
    else:  # Upload File
        uploaded_file = st.file_uploader(
            "Choose a text file to upload:",
            type=['txt', 'md', 'text'],
            help="Upload a .txt or .md file containing the medical conversation"
        )
        
        if uploaded_file is not None:
            try:
                st.session_state.transcript = uploaded_file.getvalue().decode('utf-8')
                st.success(f"✅ File uploaded successfully! ({len(st.session_state.transcript)} characters)")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.session_state.transcript = ""
        else:
            st.session_state.transcript = ""
    
    # Show transcript preview if available
    if st.session_state.transcript:
        with st.expander("📄 Transcript Preview", expanded=False):
            preview_text = st.session_state.transcript[:500]
            if len(st.session_state.transcript) > 500:
                preview_text += "...\n\n[Full transcript: " + str(len(st.session_state.transcript)) + " characters]"
            st.text(preview_text)
    
    st.divider()
    
    # Processing section with icon - ADD THIS SECTION
    st.markdown('<div class="section-with-icon"><span>⚡</span><span>Processing</span></div>', unsafe_allow_html=True)
    
    analyze_disabled = not st.session_state.transcript.strip()
    analyze_button = st.button(
        "🚀 **Analyze Transcript**",
        type="primary" if not analyze_disabled else "secondary",
        use_container_width=True,
        disabled=analyze_disabled,
        help="Click to analyze the transcript" if not analyze_disabled else "Please provide a transcript first"
    )
    
    if analyze_disabled:
        st.caption("⏸️ Enter or upload a transcript to enable analysis")
    
    st.divider()
    
    # Export Results section with icon - ADD THIS SECTION
    st.markdown('<div class="section-with-icon"><span>📊</span><span>Export Results</span></div>', unsafe_allow_html=True)
    
    if st.session_state.results and st.session_state.results.get("processing_success"):
        json_data = json.dumps(st.session_state.results, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="💾 Download JSON Report",
            data=json_data,
            file_name=f"medical_analysis_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        # Also show save location
        if 'last_saved' in st.session_state:
            st.caption(f"Last saved: {st.session_state.last_saved}")
            

# Main content area
if analyze_button and st.session_state.transcript.strip():
    with st.spinner("🔍 **Analyzing medical transcript...** This may take a moment."):
        try:
            st.session_state.results = st.session_state.pipeline.process_transcript(st.session_state.transcript)
            
            # Save results to file
            if st.session_state.results.get("processing_success"):
                saved_file = st.session_state.pipeline.save_results(st.session_state.results)
                st.session_state.last_saved = saved_file
            
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
            st.session_state.results = {
                "timestamp": datetime.now().isoformat(),
                "processing_success": False,
                "error": str(e)
            }

# Display results
if st.session_state.results:
    results = st.session_state.results
    
    if results.get("processing_success", False):
        st.success("✅ **Analysis Complete!** Results are ready below.")
        
        # Display metrics
        st.markdown('<h2 class="section-header">📈 Analysis Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            turns = len(results.get('dialogue_turns', []))
            st.metric("Dialogue Turns", turns, help="Number of conversation exchanges")
        
        with col2:
            symptoms = len(results.get('medical_summary', {}).get('Symptoms', []))
            st.metric("Symptoms Found", symptoms, help="Unique symptoms identified")
        
        with col3:
            sentiment = results.get('sentiment_analysis', {}).get('sentiment', 'UNKNOWN')
            confidence = results.get('sentiment_analysis', {}).get('confidence', 0)
            st.metric("Patient Sentiment", sentiment, delta=f"{confidence:.0%} confidence")
        
        with col4:
            entities = sum(len(v) for v in results.get('extracted_entities', {}).values())
            st.metric("Medical Entities", entities, help="Total medical terms extracted")
        
        # Tabs for detailed results
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏥 **Medical Summary**", 
            "😊 **Sentiment Analysis**", 
            "📋 **SOAP Note**", 
            "🔍 **Raw Data**"
        ])
        
        with tab1:
            summary = results.get('medical_summary', {})
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Patient Information with icon
                st.markdown("""
                <div class="section-with-icon">
                    <span>👤</span>
                    <span>Patient Information</span>
                </div>
                """, unsafe_allow_html=True)
                
                patient_name = summary.get('Patient_Name', 'Not specified in transcript')
                st.markdown(f"""
                <div class="white-box">
                    <div class="white-box-title">
                        <span>👤</span>
                        <span>Patient Details</span>
                    </div>
                    <p style="color: #1F2937;"><strong>Name:</strong> {patient_name}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Symptoms Identified with icon
                st.markdown("""
                <div class="section-with-icon">
                    <span>🩺</span>
                    <span>Symptoms Identified</span>
                </div>
                """, unsafe_allow_html=True)
                
                symptoms_list = summary.get('Symptoms', [])
                if symptoms_list:
                    symptoms_html = '<div class="white-box">'
                    symptoms_html += '<div class="white-box-title"><span>📋</span><span>Reported Symptoms</span></div>'
                    for symptom in symptoms_list[:15]:  # Limit display
                        symptoms_html += f'<span class="symptom-badge">{symptom}</span>'
                    symptoms_html += '</div>'
                    st.markdown(symptoms_html, unsafe_allow_html=True)
                    
                    if len(symptoms_list) > 15:
                        st.caption(f"... and {len(symptoms_list) - 15} more symptoms")
                else:
                    st.markdown(f"""
                    <div class="white-box">
                        <div class="white-box-title"><span>⚠️</span><span>No Symptoms Identified</span></div>
                        <p style="color: #6B7280;">No specific symptoms were identified in the transcript</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Diagnosis with icon
                st.markdown("""
                <div class="section-with-icon">
                    <span>🏥</span>
                    <span>Diagnosis</span>
                </div>
                """, unsafe_allow_html=True)
                
                diagnosis = summary.get('Diagnosis', 'Not specified')
                diagnosis_color = "#10B981" if diagnosis != "Condition requiring evaluation" else "#F59E0B"
                diagnosis_icon = "✅" if diagnosis != "Condition requiring evaluation" else "⚠️"
                
                st.markdown(f"""
                <div class="white-box">
                    <div class="white-box-title">
                        <span>{diagnosis_icon}</span>
                        <span style="color: {diagnosis_color};">Clinical Assessment</span>
                    </div>
                    <p style="color: #1F2937; font-size: 1.1rem; font-weight: 600;">{diagnosis}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Treatments Mentioned with icon
                st.markdown("""
                <div class="section-with-icon">
                    <span>💊</span>
                    <span>Treatments Mentioned</span>
                </div>
                """, unsafe_allow_html=True)
                
                treatments = summary.get('Treatment', [])
                if treatments:
                    treatments_html = '<div class="white-box">'
                    treatments_html += '<div class="white-box-title"><span>💊</span><span>Treatment Options</span></div>'
                    for treatment in treatments:
                        treatments_html += f'<p style="color: #1F2937; margin: 0.5rem 0;">• {treatment}</p>'
                    treatments_html += '</div>'
                    st.markdown(treatments_html, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="white-box">
                        <div class="white-box-title"><span>ℹ️</span><span>No Treatments Mentioned</span></div>
                        <p style="color: #6B7280;">No specific treatments were mentioned in the conversation</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Timeline with icon
                st.markdown("""
                <div class="section-with-icon">
                    <span>📅</span>
                    <span>Timeline</span>
                </div>
                """, unsafe_allow_html=True)
                
                timeline = summary.get('Timeline', [])
                if timeline:
                    timeline_html = '<div class="white-box">'
                    timeline_html += '<div class="white-box-title"><span>📅</span><span>Event Timeline</span></div>'
                    for i, event in enumerate(timeline[:5]):
                        timeline_html += f'<p style="color: #1F2937; margin: 0.5rem 0;">{i+1}. {event}</p>'
                    timeline_html += '</div>'
                    st.markdown(timeline_html, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="white-box">
                        <div class="white-box-title"><span>ℹ️</span><span>No Timeline Events</span></div>
                        <p style="color: #6B7280;">No timeline events were extracted from the transcript</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Summary with icon
                st.markdown("""
                <div class="section-with-icon">
                    <span>📝</span>
                    <span>Summary</span>
                </div>
                """, unsafe_allow_html=True)
                
                summary_text = summary.get('Summary', 'No summary generated')
                st.markdown(f"""
                <div class="white-box">
                    <div class="white-box-title"><span>📋</span><span>Clinical Summary</span></div>
                    <p style="color: #1F2937;">{summary_text}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            sentiment_data = results.get('sentiment_analysis', {})
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("##### 🎭 Sentiment Analysis")
                sentiment = sentiment_data.get('sentiment', 'UNKNOWN')
                confidence = sentiment_data.get('confidence', 0)
                
                # Color coding for sentiment
                sentiment_config = {
                    'ANXIOUS': {'color': '#EF4444', 'icon': '😟', 'desc': 'Patient shows signs of worry or anxiety'},
                    'NEUTRAL': {'color': '#6B7280', 'icon': '😐', 'desc': 'Patient maintains neutral emotional state'},
                    'REASSURED': {'color': '#10B981', 'icon': '😊', 'desc': 'Patient appears reassured or relieved'},
                    'PAINFUL': {'color': '#F59E0B', 'icon': '🤕', 'desc': 'Patient expresses pain or discomfort'}
                }
                
                config = sentiment_config.get(sentiment, {'color': '#6B7280', 'icon': '❓', 'desc': 'Unknown sentiment'})
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem; text-align: center; margin-bottom: 0.5rem;">
                        {config['icon']}
                    </div>
                    <h3 style="color: {config['color']}; text-align: center; margin: 0;">
                        {sentiment}
                    </h3>
                    <p style="text-align: center; color: #6B7280; margin: 0.5rem 0 0 0;">
                        Confidence: {confidence:.0%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.caption(config['desc'])
                
                st.markdown("##### 🎯 Patient Intent")
                intent = sentiment_data.get('intent', 'Not determined')
                st.markdown(f'<div class="info-box"><strong>{intent}</strong></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("##### 💬 Patient Dialogue Extract")
                patient_text = results.get('patient_text', 'No patient dialogue extracted')
                
                if patient_text:
                    # Display with scrollable area
                    st.text_area(
                        "Extracted patient statements:",
                        value=patient_text,
                        height=300,
                        disabled=True,
                                                label_visibility="collapsed"
                    )
                    
                    st.caption(f"Extracted {len(patient_text.split())} words of patient dialogue")
                else:
                    st.warning("No patient dialogue could be extracted from the transcript")
                
                st.markdown("##### 📊 Sentiment Breakdown")
                scores = sentiment_data.get('scores', {})
                if scores:
                    import plotly.express as px
                    import pandas as pd
                    
                    # Create bar chart
                    df = pd.DataFrame(list(scores.items()), columns=['Sentiment', 'Count'])
                    df = df.sort_values('Count', ascending=True)
                    
                    fig = px.bar(df, x='Count', y='Sentiment', orientation='h',
                                 color='Count', color_continuous_scale='Blues',
                                 title='Sentiment Frequency in Patient Dialogue')
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            soap_note = results.get('soap_note', {})
            
            st.markdown("##### 📋 SOAP Clinical Note")
            st.info("Generated structured clinical note following SOAP format")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Subjective Section
                st.markdown("**Subjective (S)**")
                subjective = soap_note.get('Subjective', {})
                
                st.markdown(f"""
                <div class="info-box">
                    <strong style="color: #1E40AF;">Chief Complaint:</strong><br>
                    <span style="color: #1E40AF;">{subjective.get('Chief_Complaint', 'Not specified')}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-box">
                    <strong style="color: #1E40AF;">History of Present Illness:</strong><br>
                    <span style="color: #1E40AF;">{subjective.get('History_of_Present_Illness', 'Not specified')}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Objective Section
                st.markdown("**Objective (O)**")
                objective = soap_note.get('Objective', {})
                
                for key, value in objective.items():
                    if key != '_raw':
                        st.markdown(f"""
                        <div style="padding: 0.5rem 0; border-bottom: 1px solid #E5E7EB;">
                            <strong style="color: #4B5563;">{key.replace('_', ' ')}:</strong> 
                            <span style="color: #1F2937;">{value}</span>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                # Assessment Section
                st.markdown("**Assessment (A)**")
                assessment = soap_note.get('Assessment', {})
                
                # Diagnosis with better styling
                diagnosis_text = assessment.get('Diagnosis', 'Not specified')
                st.markdown(f"""
                <div class="soap-diagnosis">
                    <strong style="color: #92400E;">Diagnosis:</strong><br>
                    <span style="color: #92400E; font-size: 1.1rem; font-weight: 600;">{diagnosis_text}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Severity and Prognosis
                st.markdown(f"""
                <div class="info-box">
                    <strong style="color: #1E40AF;">Severity:</strong> 
                    <span style="color: #1E40AF;">{assessment.get('Severity', 'Not specified')}</span><br>
                    <strong style="color: #1E40AF;">Prognosis:</strong> 
                    <span style="color: #1E40AF;">{assessment.get('Prognosis', 'Not specified')}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Plan Section
                st.markdown("**Plan (P)**")
                plan = soap_note.get('Plan', {})
                
                st.markdown("**Treatment Plan:**")
                treatments = plan.get('Treatment', [])
                if isinstance(treatments, list):
                    for treatment in treatments:
                        st.markdown(f"• <span style='color: #1F2937;'>{treatment}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"• <span style='color: #1F2937;'>{treatments}</span>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="padding: 0.5rem 0;">
                    <strong style="color: #4B5563;">Follow-up:</strong> 
                    <span style="color: #1F2937;">{plan.get('Follow_Up', 'Not specified')}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Medications:**")
                meds = plan.get('Medications', [])
                for med in meds:
                    st.markdown(f"• <span style='color: #1F2937;'>{med}</span>", unsafe_allow_html=True)
                
                # Plan Section
                st.markdown("**Plan (P)**")
                plan = soap_note.get('Plan', {})
                
                st.markdown("**Treatment Plan:**")
                treatments = plan.get('Treatment', [])
                if isinstance(treatments, list):
                    for treatment in treatments:
                        st.markdown(f"• {treatment}")
                else:
                    st.markdown(f"• {treatments}")
                
                st.markdown(f"**Follow-up:** {plan.get('Follow_Up', 'Not specified')}")
                
                st.markdown("**Medications:**")
                meds = plan.get('Medications', [])
                for med in meds:
                    st.markdown(f"• {med}")
            
            # SOAP JSON View
            with st.expander("📄 View SOAP Note Structure", expanded=False):
                st.json(soap_note)
        
        with tab4:
            st.markdown("##### 🔍 Raw Analysis Data")
            
            # Entities section
            st.markdown("**📌 Extracted Medical Entities**")
            entities = results.get('extracted_entities', {})
            
            for category, items in entities.items():
                if items:
                    st.markdown(f"**{category.replace('_', ' ').title()}**")
                    
                    # Create badges for entities
                    badges_html = ""
                    for item in items:
                        badges_html += f'<span class="symptom-badge">{item}</span>'
                    
                    if badges_html:
                        st.markdown(badges_html, unsafe_allow_html=True)
            
            # Keywords section
            st.markdown("**🔑 Top Keywords**")
            keywords = results.get('keywords', [])
            if keywords:
                col_keywords = st.columns(4)
                for i, keyword in enumerate(keywords[:16]):  # Show up to 16 keywords
                    with col_keywords[i % 4]:
                        st.markdown(f'<div class="symptom-badge">{keyword}</div>', unsafe_allow_html=True)
            
            # Raw JSON
            with st.expander("📋 View Complete Raw JSON", expanded=False):
                st.json(results)
            
            # Dialogue turns
            st.markdown("**💬 Structured Dialogue**")
            dialogue_turns = results.get('dialogue_turns', [])
            if dialogue_turns:
                # Show first few turns
                display_turns = dialogue_turns[:8]
                for i, turn in enumerate(display_turns):
                    speaker_color = "#3B82F6" if "physician" in turn['speaker'].lower() else "#10B981"
                    st.markdown(f"""
                    <div style="background-color: {'#EFF6FF' if i % 2 == 0 else '#F9FAFB'}; 
                                padding: 10px; border-radius: 8px; margin-bottom: 5px;">
                        <span style="color: {speaker_color}; font-weight: bold;">
                            {turn['speaker']}:
                        </span> 
                        <span style="color: #4B5563;">
                            {turn['text'][:150]}{'...' if len(turn['text']) > 150 else ''}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(dialogue_turns) > 8:
                    st.caption(f"... and {len(dialogue_turns) - 8} more dialogue turns")
    
    else:
        # Error case
        st.error("❌ **Analysis Failed**")
        st.error(f"Error: {results.get('error', 'Unknown error')}")
        
        if st.button("🔄 Retry Analysis"):
            st.rerun()

else:
    # Welcome/instruction screen
    if not st.session_state.results:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center;">
            <h2 style="color: white; margin-bottom: 1rem;">🚀 Welcome to Medical NLP Pipeline</h2>
            <p style="font-size: 1.1rem;">
                Upload or paste a medical transcript to extract symptoms, analyze sentiment, 
                generate clinical notes, and create structured medical summaries.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ## 📋 How to Use:
        
        1. **Choose Input Method** (in the sidebar):
           - Use a sample medical transcript
           - Paste your own doctor-patient conversation
           - Upload a text file
        
        2. **Click "🚀 Analyze Transcript"** to process
        
        3. **Review Results** across four tabs:
           - 🏥 Medical Summary
           - 😊 Sentiment Analysis  
           - 📋 SOAP Clinical Note
           - 🔍 Raw Data
        
        ## 🔧 Features:
        
        | Feature | Description |
        |---------|-------------|
        | **Medical NER** | Extracts symptoms, diagnoses, treatments, body parts |
        | **Sentiment Analysis** | Analyzes patient emotional state and intent |
        | **SOAP Notes** | Generates structured clinical documentation |
        | **Keyword Extraction** | Identifies key medical terms |
        | **Dialogue Parsing** | Structures conversation into turns |
        
        ## 📁 Sample Formats:
        
        ```
        **Physician:** How are you feeling today?
        **Patient:** I'm experiencing back pain since the accident.
        **Physician:** Can you describe the pain?
        **Patient:** It's a dull ache that worsens when I sit for long periods.
        ```
        
        ---
        
        ⚕️ **Disclaimer**: This tool is for educational and research purposes only. 
        It does not provide medical advice, diagnosis, or treatment recommendations.
        """)
        
        # Quick start examples
        st.markdown("### 🚀 Quick Start")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Try Car Accident Sample", use_container_width=True):
                st.session_state.sample_option = "Car Accident & Whiplash"
                st.session_state.input_method = "📝 Sample Transcript"
                st.rerun()
        
        with col2:
            if st.button("🩺 Try Chronic Pain Sample", use_container_width=True):
                st.session_state.sample_option = "Chronic Back Pain"
                st.session_state.input_method = "📝 Sample Transcript"
                st.rerun()
        
        with col3:
            if st.button("📋 Try Post-Surgery Sample", use_container_width=True):
                st.session_state.sample_option = "Post-Surgery Follow-up"
                st.session_state.input_method = "📝 Sample Transcript"
                st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem;">
    <p>⚕️ Medical NLP Pipeline v1.0 • For educational and research purposes only</p>
    <p>⚠️ <strong>Not for clinical decision-making</strong> • Consult healthcare professionals for medical advice</p>
</div>
""", unsafe_allow_html=True)