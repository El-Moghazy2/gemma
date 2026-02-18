"""Test DDInter API integration."""

import logging
logging.basicConfig(level=logging.DEBUG)

from healthpost.ddinter_api import DDInterClient

def test_ddinter():
    print("Testing DDInter API integration...\n")

    client = DDInterClient()

    # Test 1: Search for drugs
    print("1. Searching for 'Omeprazole'...")
    omeprazole_id = client.search_drug("Omeprazole")
    print(f"   Result: {omeprazole_id}")

    print("\n2. Searching for 'Fosphenytoin'...")
    fosphenytoin_id = client.search_drug("Fosphenytoin")
    print(f"   Result: {fosphenytoin_id}")

    print("\n3. Searching for 'Metformin'...")
    metformin_id = client.search_drug("Metformin")
    print(f"   Result: {metformin_id}")

    # Test 4: Check interactions between drugs with known interaction
    print("\n4. Checking interactions between Omeprazole and Fosphenytoin...")
    drugs = ["Omeprazole", "Fosphenytoin"]
    interactions = client.check_interactions(drugs)

    if interactions:
        print(f"   Found {len(interactions)} interactions:")
        for interaction in interactions:
            print(f"   - {interaction.drug1} + {interaction.drug2}")
            print(f"     Severity: {interaction.severity}")
            print(f"     Description: {interaction.description[:100]}...")
            print(f"     Management: {interaction.management[:100]}...")
    else:
        print("   No interactions found")

    # Test 5: Check interactions between multiple drugs
    print("\n5. Checking interactions between Omeprazole, Fosphenytoin, and Citalopram...")
    drugs = ["Omeprazole", "Fosphenytoin", "Citalopram"]
    interactions = client.check_interactions(drugs)

    if interactions:
        print(f"   Found {len(interactions)} interactions:")
        for interaction in interactions:
            print(f"   - {interaction.drug1} + {interaction.drug2}: {interaction.severity}")
    else:
        print("   No interactions found")

    print("\n[DONE] DDInter API test complete")

if __name__ == "__main__":
    test_ddinter()
