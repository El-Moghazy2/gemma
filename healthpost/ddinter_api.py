"""
DDInter API client for drug interaction checking.

DDInter is a comprehensive drug-drug interaction database maintained by
SCBDD (South China Bioinformatics Data Laboratory).
Contains 302,516 DDI associations between 2,290 drugs.

Website: https://ddinter2.scbdd.com/
"""

import json
import logging
import urllib.request
import urllib.error
import urllib.parse
import ssl
from typing import List, Optional, Dict
from dataclasses import dataclass
import time
import re

logger = logging.getLogger(__name__)

# API endpoints
DDINTER_BASE = "https://ddinter2.scbdd.com"

# Common drug name aliases (brand/common name -> generic name used in DDInter)
DRUG_NAME_ALIASES = {
    "aspirin": "acetylsalicylic acid",
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen",
    "prilosec": "omeprazole",
    "nexium": "esomeprazole",
    "lipitor": "atorvastatin",
    "zocor": "simvastatin",
    "plavix": "clopidogrel",
    "coumadin": "warfarin",
    "glucophage": "metformin",
    "zoloft": "sertraline",
    "prozac": "fluoxetine",
    "lexapro": "escitalopram",
    "celexa": "citalopram",
    "xanax": "alprazolam",
    "valium": "diazepam",
    "ativan": "lorazepam",
    "ambien": "zolpidem",
    "viagra": "sildenafil",
    "cialis": "tadalafil",
    "synthroid": "levothyroxine",
    "lasix": "furosemide",
    "zantac": "ranitidine",
    "pepcid": "famotidine",
    "benadryl": "diphenhydramine",
    "zyrtec": "cetirizine",
    "claritin": "loratadine",
    "allegra": "fexofenadine",
    "flagyl": "metronidazole",
    "cipro": "ciprofloxacin",
    "amoxil": "amoxicillin",
    "augmentin": "amoxicillin",
    "zithromax": "azithromycin",
    "prednisone": "prednisolone",
}


@dataclass
class DDInterInteraction:
    """Drug interaction from DDInter API."""
    drug1: str
    drug2: str
    severity: str  # "Major", "Moderate", "Minor", "Unknown"
    description: str
    management: str


class DDInterClient:
    """
    Client for DDInter Drug Interaction API.

    DDInter contains 302,516 drug-drug interactions compiled from
    DrugBank, KEGG, and other authoritative sources.
    """

    def __init__(self):
        self.base_url = DDINTER_BASE
        self._drug_cache: Dict[str, Optional[str]] = {}  # drug name -> DDInter ID
        self._name_cache: Dict[str, str] = {}  # DDInter ID -> drug name
        self._last_request_time = 0
        self._min_request_interval = 0.2  # Be respectful of the server

        # Create SSL context for HTTPS
        self._ssl_context = ssl.create_default_context()

    def _rate_limit(self):
        """Ensure we don't overload the server."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _request_get(self, endpoint: str) -> Optional[str]:
        """Make a GET request and return content."""
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "HealthPost/1.0 (Medical Decision Support)")
            req.add_header("Accept", "text/html,application/json")

            with urllib.request.urlopen(req, timeout=15, context=self._ssl_context) as response:
                return response.read().decode('utf-8')
        except urllib.error.URLError as e:
            logger.warning(f"DDInter API request failed for {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"DDInter API error: {e}")
            return None

    def search_drug(self, drug_name: str) -> Optional[str]:
        """
        Search for a drug and get its DDInter ID.

        Args:
            drug_name: Drug name to search for (min 3 characters)

        Returns:
            DDInter ID (e.g., "DDInter1340") or None if not found
        """
        # Normalize and check cache
        cache_key = drug_name.lower().strip()
        if cache_key in self._drug_cache:
            return self._drug_cache[cache_key]

        # Check for common name aliases
        search_term = DRUG_NAME_ALIASES.get(cache_key, drug_name.strip())

        # Need at least 3 characters for DDInter search
        if len(search_term) < 3:
            search_term = search_term + "   "  # Pad if too short

        # Use the check-datasource endpoint
        endpoint = f"check-datasource/{urllib.parse.quote(search_term[:10])}/"
        response = self._request_get(endpoint)

        if not response:
            self._drug_cache[cache_key] = None
            return None

        try:
            data = json.loads(response)
            results = data.get("data", [])

            if results:
                # Find best match (exact match preferred)
                for result in results:
                    if result.get("name", "").lower() == cache_key:
                        ddinter_id = result.get("internalID")
                        self._drug_cache[cache_key] = ddinter_id
                        self._name_cache[ddinter_id] = result.get("name")
                        logger.debug(f"Found exact DDInter ID for {drug_name}: {ddinter_id}")
                        return ddinter_id

                # No exact match, use first result
                first_result = results[0]
                ddinter_id = first_result.get("internalID")
                self._drug_cache[cache_key] = ddinter_id
                self._name_cache[ddinter_id] = first_result.get("name")
                logger.debug(f"Found DDInter ID for {drug_name}: {ddinter_id} ({first_result.get('name')})")
                return ddinter_id

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse DDInter search response for {drug_name}")

        self._drug_cache[cache_key] = None
        logger.debug(f"Drug not found in DDInter: {drug_name}")
        return None

    def check_interactions(self, drug_names: List[str]) -> List[DDInterInteraction]:
        """
        Check for interactions between a list of drugs using DDInter checker.

        Args:
            drug_names: List of drug names to check (2-5 drugs)

        Returns:
            List of interactions found
        """
        if len(drug_names) < 2:
            return []

        # Get DDInter IDs for all drugs
        ddinter_ids = []
        id_to_name = {}

        for name in drug_names:
            ddinter_id = self.search_drug(name)
            if ddinter_id:
                ddinter_ids.append(ddinter_id)
                # Use cached name if available, otherwise original name
                id_to_name[ddinter_id] = self._name_cache.get(ddinter_id, name)
            else:
                logger.warning(f"Could not find DDInter ID for: {name}")

        if len(ddinter_ids) < 2:
            logger.warning("Not enough drugs found in DDInter to check interactions")
            return []

        # Limit to 5 drugs as per DDInter checker
        if len(ddinter_ids) > 5:
            logger.warning("DDInter checker supports max 5 drugs, truncating list")
            ddinter_ids = ddinter_ids[:5]

        # Get the checker results page
        ids_str = "-".join(ddinter_ids)
        endpoint = f"checker/result/{ids_str}/"
        html = self._request_get(endpoint)

        if not html:
            return []

        # Parse interaction results from the page
        interactions = self._parse_checker_results(html, id_to_name)

        return interactions

    def _parse_checker_results(
        self, html: str, id_to_name: Dict[str, str]
    ) -> List[DDInterInteraction]:
        """Parse interaction results from checker result page."""
        interactions = []

        # Look for response_data JavaScript variable in the page
        # Pattern: let response_data = [...];
        pattern = r"let\s+response_data\s*=\s*(\[.*?\]);"
        match = re.search(pattern, html, re.DOTALL)

        if not match:
            logger.warning("Could not find response_data in DDInter result page")
            return []

        try:
            # Parse the JavaScript array as JSON
            # Clean up any JavaScript-specific syntax
            json_str = match.group(1)
            # Replace single quotes with double quotes for valid JSON
            json_str = json_str.replace("'", '"')

            data = json.loads(json_str)

            for item in data:
                drug1 = item.get("drug_a_name", "Unknown")
                drug2 = item.get("drug_b_name", "Unknown")
                severity = item.get("idx__level", "Unknown")
                description = item.get("idx__interaction_description", "-")
                management = item.get("idx__management", "-")

                # Skip if no actual interaction data
                if description == "-" and severity == "Unknown":
                    continue

                interactions.append(DDInterInteraction(
                    drug1=drug1,
                    drug2=drug2,
                    severity=severity,
                    description=description if description != "-" else f"Potential interaction between {drug1} and {drug2}",
                    management=management if management != "-" else "Consult prescriber for guidance."
                ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse DDInter response_data: {e}")
        except Exception as e:
            logger.error(f"Error parsing DDInter results: {e}")

        return interactions


# Singleton instance
_client: Optional[DDInterClient] = None


def get_ddinter_client() -> DDInterClient:
    """Get or create DDInter client instance."""
    global _client
    if _client is None:
        _client = DDInterClient()
    return _client


def check_interactions_online(drug_names: List[str]) -> List[DDInterInteraction]:
    """
    Convenience function to check drug interactions via DDInter API.

    Args:
        drug_names: List of drug names

    Returns:
        List of interactions
    """
    client = get_ddinter_client()
    return client.check_interactions(drug_names)
