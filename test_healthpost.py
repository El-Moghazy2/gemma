"""
Test script for HealthPost components.

Run with: python test_healthpost.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_drug_database():
    """Test the drug database functionality."""
    print("\n" + "=" * 50)
    print("Testing Drug Database")
    print("=" * 50)

    from healthpost.drugs import DrugDatabase

    db = DrugDatabase()

    # Test drug lookup
    print("\n1. Testing drug lookup...")
    info = db.get_drug_info("paracetamol")
    if info:
        print(f"   [OK] Found: {info.name} ({info.generic_name})")
        print(f"     Class: {info.drug_class}")
        print(f"     Uses: {', '.join(info.common_uses[:3])}")
    else:
        print("   [FAIL] Drug not found")

    # Test interaction checking
    print("\n2. Testing interaction checking...")
    meds = ["Metformin", "Alcohol"]
    interactions = db.check_interactions(meds)
    if interactions:
        print(f"   [OK] Found {len(interactions)} interaction(s):")
        for inter in interactions:
            print(f"     - {inter.drugs[0]} + {inter.drugs[1]}: {inter.severity}")
    else:
        print("   [FAIL] No interactions found (expected 1)")

    # Test no interaction
    print("\n3. Testing safe combination...")
    safe_meds = ["Paracetamol", "Amoxicillin"]
    safe_interactions = db.check_interactions(safe_meds)
    if not safe_interactions:
        print(f"   [OK] No interactions (as expected)")
    else:
        print(f"   [FAIL] Unexpected interactions found")

    db.close()
    print("\n   Drug database tests complete!")


def test_triage_agent():
    """Test the triage agent functionality."""
    print("\n" + "=" * 50)
    print("Testing Triage Agent")
    print("=" * 50)

    from healthpost.triage import TriageAgent
    from healthpost.config import Config

    config = Config()
    agent = TriageAgent(config)

    # Test malaria scenario
    print("\n1. Testing malaria diagnosis...")
    symptoms = "Patient has high fever for 3 days with severe headache"
    diagnosis, treatment = agent.diagnose_and_treat(symptoms, [])

    print(f"   Diagnosis: {diagnosis.condition}")
    print(f"   Confidence: {diagnosis.confidence:.0%}")
    print(f"   Medications: {[m.name for m in treatment.medications]}")

    # Test diarrhea scenario
    print("\n2. Testing gastroenteritis diagnosis...")
    symptoms = "Child has watery diarrhea for 2 days"
    diagnosis, treatment = agent.diagnose_and_treat(symptoms, [])

    print(f"   Diagnosis: {diagnosis.condition}")
    print(f"   Confidence: {diagnosis.confidence:.0%}")
    print(f"   Medications: {[m.name for m in treatment.medications]}")

    # Test with visual findings
    print("\n3. Testing with visual findings...")
    symptoms = "Patient has itchy skin"
    visual = ["Circular rash with scaling", "Erythematous borders"]
    diagnosis, treatment = agent.diagnose_and_treat(symptoms, visual)

    print(f"   Diagnosis: {diagnosis.condition}")
    print(f"   Evidence: {diagnosis.supporting_evidence[:2]}")

    print("\n   Triage agent tests complete!")


def test_complete_workflow():
    """Test the complete HealthPost workflow."""
    print("\n" + "=" * 50)
    print("Testing Complete Workflow")
    print("=" * 50)

    from healthpost.core import HealthPost
    from healthpost.config import Config

    # Use mock mode for testing without models
    config = Config()
    hp = HealthPost(config)

    print("\n1. Testing patient visit (text only)...")
    result = hp.patient_visit(
        symptoms_text="Patient has high fever for 3 days with rash on trunk",
        existing_meds_list=["Metformin"],
    )

    print(f"   Diagnosis: {result.diagnosis.condition}")
    print(f"   Confidence: {result.overall_confidence:.0%}")
    print(f"   Safe to proceed: {result.is_safe_to_proceed}")
    print(f"   Needs referral: {result.needs_referral}")

    print("\n2. Testing formatted output...")
    output = result.format_for_display()
    print(output[:500] + "..." if len(output) > 500 else output)

    print("\n   Complete workflow tests complete!")


def test_voice_mock():
    """Test voice transcription in mock mode."""
    print("\n" + "=" * 50)
    print("Testing Voice Transcriber (Mock)")
    print("=" * 50)

    from healthpost.voice import VoiceTranscriber
    from healthpost.config import Config
    import numpy as np

    config = Config()
    voice = VoiceTranscriber(config)

    # Create mock audio (short)
    print("\n1. Testing short audio...")
    short_audio = np.zeros(config.sample_rate // 2)  # 0.5 seconds
    result = voice.transcribe(short_audio)
    print(f"   Result: {result}")

    # Create mock audio (medium)
    print("\n2. Testing medium audio...")
    medium_audio = np.zeros(config.sample_rate * 3)  # 3 seconds
    result = voice.transcribe(medium_audio)
    print(f"   Result: {result}")

    print("\n   Voice transcriber tests complete!")


def test_vision_mock():
    """Test vision analyzer in mock mode."""
    print("\n" + "=" * 50)
    print("Testing Vision Analyzer (Mock)")
    print("=" * 50)

    from healthpost.vision import MedicalVisionAnalyzer
    from healthpost.config import Config
    import numpy as np

    config = Config()
    vision = MedicalVisionAnalyzer(config)

    # Create a mock image
    print("\n1. Testing skin analysis...")
    mock_image = np.zeros((224, 224, 3), dtype=np.uint8)
    result = vision.analyze_skin_condition(mock_image)
    print(f"   Raw analysis: {result.get('raw_analysis', '')[:100]}...")

    print("\n2. Testing medication extraction...")
    result = vision.extract_medications(mock_image)
    print(f"   Medications found: {result}")

    print("\n   Vision analyzer tests complete!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  HEALTHPOST TEST SUITE")
    print("=" * 60)

    try:
        test_drug_database()
    except Exception as e:
        print(f"\n   [FAIL] Drug database test failed: {e}")

    try:
        test_triage_agent()
    except Exception as e:
        print(f"\n   [FAIL] Triage agent test failed: {e}")

    try:
        test_voice_mock()
    except Exception as e:
        print(f"\n   [FAIL] Voice test failed: {e}")

    try:
        test_vision_mock()
    except Exception as e:
        print(f"\n   [FAIL] Vision test failed: {e}")

    try:
        test_complete_workflow()
    except Exception as e:
        print(f"\n   [FAIL] Complete workflow test failed: {e}")

    print("\n" + "=" * 60)
    print("  TEST SUITE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
