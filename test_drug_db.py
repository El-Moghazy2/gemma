"""Test DrugDatabase with DDInter API integration."""

import logging
logging.basicConfig(level=logging.INFO)

from healthpost.drugs import DrugDatabase

def test_drug_database():
    print("Testing DrugDatabase with DDInter API...\n")

    db = DrugDatabase()

    # Test 1: Check interactions between drugs with known interaction
    print("1. Checking interactions between Omeprazole and Fosphenytoin...")
    interactions = db.check_interactions(["Omeprazole", "Fosphenytoin"])

    if interactions:
        print(f"   Found {len(interactions)} interactions:")
        for interaction in interactions:
            print(f"   - {interaction.drugs[0]} + {interaction.drugs[1]}")
            print(f"     Severity: {interaction.severity}")
            print(f"     Description: {interaction.description[:80]}...")
            print(f"     Recommendation: {interaction.recommendation[:80]}...")
    else:
        print("   No interactions found")

    # Test 2: Check interactions between multiple drugs
    print("\n2. Checking interactions between Warfarin, Aspirin, and Ibuprofen...")
    interactions = db.check_interactions(["Warfarin", "Aspirin", "Ibuprofen"])

    if interactions:
        print(f"   Found {len(interactions)} interactions:")
        for interaction in interactions:
            print(f"   - {interaction.drugs[0]} + {interaction.drugs[1]}: {interaction.severity}")
    else:
        print("   No interactions found")

    # Test 3: Get drug info (from local DB)
    print("\n3. Getting drug info for Metformin (from local DB)...")
    info = db.get_drug_info("Metformin")
    if info:
        print(f"   Name: {info.name}")
        print(f"   Generic: {info.generic_name}")
        print(f"   Class: {info.drug_class}")
        print(f"   Uses: {', '.join(info.common_uses)}")
    else:
        print("   Drug not found in local DB")

    print("\n[DONE] DrugDatabase test complete")

if __name__ == "__main__":
    test_drug_database()
