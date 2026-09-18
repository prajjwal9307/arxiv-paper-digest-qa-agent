from app.prompts import QA_SYSTEM

def test_grounding_instruction():
    assert "I couldn't find this information in the paper." in QA_SYSTEM
