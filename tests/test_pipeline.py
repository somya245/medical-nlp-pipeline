"""
Test package for Medical NLP Pipeline
"""

__all__ = ['test_pipeline']

# Test data for use in other test modules
TEST_TRANSCRIPTS = {
    'minimal': """
    Physician: How are you?
    Patient: I have back pain.
    Physician: How long?
    Patient: Two weeks.
    """,
    
    'whiplash': """
    Physician: Good morning. What brings you in today?
    Patient: I was in a car accident last week and my neck hurts.
    Physician: I see. Can you describe the pain?
    Patient: It's a sharp pain when I turn my head.
    Physician: Were you diagnosed with anything at the emergency room?
    Patient: They said it might be whiplash.
    """,
    
    'recovery': """
    Physician: How has your recovery been?
    Patient: Much better, thank you. The physiotherapy really helped.
    Physician: That's great to hear. Any residual pain?
    Patient: Just a little stiffness in the morning, but it goes away quickly.
    Physician: Excellent progress.
    """
}

# Expected outputs for validation
EXPECTED_OUTPUTS = {
    'medical_summary_keys': [
        'Patient_Name',
        'Symptoms',
        'Diagnosis',
        'Treatment',
        'Current_Status',
        'Prognosis'
    ],
    'sentiment_labels': ['ANXIOUS', 'NEUTRAL', 'REASSURED', 'PAINFUL'],
    'soap_sections': ['Subjective', 'Objective', 'Assessment', 'Plan']
}

# Utility functions for tests
def create_mock_transcript(num_exchanges=5):
    """Create a mock transcript for testing"""
    transcript = ""
    for i in range(num_exchanges):
        transcript += f"Physician: Question {i}?\n"
        transcript += f"Patient: Answer {i}.\n\n"
    return transcript

def assert_dict_structure(test_case, data, expected_keys, dict_name="dictionary"):
    """Assert that a dictionary has the expected structure"""
    for key in expected_keys:
        test_case.assertIn(key, data, 
                          f"{dict_name} missing key: {key}")
        test_case.assertIsNotNone(data[key],
                                 f"{dict_name} key {key} is None")

def assert_list_non_empty(test_case, data, list_name="list"):
    """Assert that a list is not empty"""
    test_case.assertIsInstance(data, list, 
                              f"{list_name} should be a list")
    test_case.assertGreater(len(data), 0,
                           f"{list_name} should not be empty")