"""
RxNorm API client for drug interaction checking.

Uses the free NLM RxNav API to check drug interactions.
https://lhncbc.nlm.nih.gov/RxNav/APIs/InteractionAPIs.html
"""

import json
import logging
import urllib.request
import urllib.error
import urllib.parse
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# API endpoints
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"


@dataclass
class RxNormInteraction:
    """Drug interaction from RxNorm API."""
    drug1: str
    drug2: str
    severity: str  # "high", "moderate", "low", or "N/A"
    description: str
    source: str  # "DrugBank" or "ONCHigh"


class RxNormClient:
    """
    Client for RxNorm/RxNav Drug Interaction API.

    Free to use, no API key required.
    Rate limit: 20 requests/second
    """

    def __init__(self):
        self.base_url = RXNORM_BASE
        self._rxcui_cache: Dict[str, Optional[str]] = {}
        self._last_request_time = 0
        self._min_request_interval = 0.05  # 20 requests/second max

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to RxNorm API."""
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode())
        except urllib.error.URLError as e:
            logger.warning(f"RxNorm API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"RxNorm API error: {e}")
            return None

    def get_rxcui(self, drug_name: str) -> Optional[str]:
        """
        Get RxCUI (RxNorm Concept Unique Identifier) for a drug name.

        Args:
            drug_name: Drug name (brand or generic)

        Returns:
            RxCUI string or None if not found
        """
        # Check cache
        cache_key = drug_name.lower()
        if cache_key in self._rxcui_cache:
            return self._rxcui_cache[cache_key]

        # Try approximate match first (more forgiving)
        data = self._request("approximateTerm.json", {"term": drug_name, "maxEntries": 1})

        if data and "approximateGroup" in data:
            candidates = data["approximateGroup"].get("candidate", [])
            if candidates:
                rxcui = candidates[0].get("rxcui")
                self._rxcui_cache[cache_key] = rxcui
                return rxcui

        # Try exact match
        data = self._request("rxcui.json", {"name": drug_name})

        if data and "idGroup" in data:
            rxcui_list = data["idGroup"].get("rxnormId", [])
            if rxcui_list:
                rxcui = rxcui_list[0]
                self._rxcui_cache[cache_key] = rxcui
                return rxcui

        self._rxcui_cache[cache_key] = None
        return None

    def get_interactions_by_rxcui(self, rxcui: str) -> List[RxNormInteraction]:
        """
        Get all known interactions for a drug by RxCUI.

        Args:
            rxcui: RxNorm Concept Unique Identifier

        Returns:
            List of interactions
        """
        data = self._request(f"interaction/interaction.json", {"rxcui": rxcui})

        if not data:
            return []

        interactions = []

        # Parse interaction groups
        interaction_pairs = data.get("interactionTypeGroup", [])
        for group in interaction_pairs:
            source = group.get("sourceName", "Unknown")

            for itype in group.get("interactionType", []):
                for pair in itype.get("interactionPair", []):
                    # Get the two drugs
                    concepts = pair.get("interactionConcept", [])
                    if len(concepts) >= 2:
                        drug1 = concepts[0].get("minConceptItem", {}).get("name", "Unknown")
                        drug2 = concepts[1].get("minConceptItem", {}).get("name", "Unknown")

                        description = pair.get("description", "Interaction detected")
                        severity = pair.get("severity", "N/A")

                        interactions.append(RxNormInteraction(
                            drug1=drug1,
                            drug2=drug2,
                            severity=severity,
                            description=description,
                            source=source
                        ))

        return interactions

    def check_interactions(self, drug_names: List[str]) -> List[RxNormInteraction]:
        """
        Check for interactions between a list of drugs.

        Args:
            drug_names: List of drug names to check

        Returns:
            List of interactions found
        """
        if len(drug_names) < 2:
            return []

        # Get RxCUIs for all drugs
        rxcuis = []
        name_map = {}  # rxcui -> original name

        for name in drug_names:
            rxcui = self.get_rxcui(name)
            if rxcui:
                rxcuis.append(rxcui)
                name_map[rxcui] = name
            else:
                logger.warning(f"Could not find RxCUI for: {name}")

        if len(rxcuis) < 2:
            logger.warning("Not enough drugs found in RxNorm to check interactions")
            return []

        # Use the list interaction endpoint
        rxcui_str = "+".join(rxcuis)
        data = self._request("interaction/list.json", {"rxcuis": rxcui_str})

        if not data:
            return []

        interactions = []

        # Parse full interaction results
        for group in data.get("fullInteractionTypeGroup", []):
            source = group.get("sourceName", "Unknown")

            for itype in group.get("fullInteractionType", []):
                for pair in itype.get("interactionPair", []):
                    concepts = pair.get("interactionConcept", [])
                    if len(concepts) >= 2:
                        drug1 = concepts[0].get("minConceptItem", {}).get("name", "Unknown")
                        drug2 = concepts[1].get("minConceptItem", {}).get("name", "Unknown")

                        description = pair.get("description", "Interaction detected")
                        severity = pair.get("severity", "N/A")

                        interactions.append(RxNormInteraction(
                            drug1=drug1,
                            drug2=drug2,
                            severity=severity,
                            description=description,
                            source=source
                        ))

        return interactions

    def get_drug_info(self, drug_name: str) -> Optional[Dict]:
        """
        Get basic drug information from RxNorm.

        Args:
            drug_name: Drug name

        Returns:
            Dict with drug info or None
        """
        rxcui = self.get_rxcui(drug_name)
        if not rxcui:
            return None

        # Get properties
        data = self._request(f"rxcui/{rxcui}/properties.json")

        if data and "properties" in data:
            props = data["properties"]
            return {
                "rxcui": rxcui,
                "name": props.get("name"),
                "synonym": props.get("synonym"),
                "tty": props.get("tty"),  # Term type
            }

        return None


# Singleton instance
_client: Optional[RxNormClient] = None


def get_rxnorm_client() -> RxNormClient:
    """Get or create RxNorm client instance."""
    global _client
    if _client is None:
        _client = RxNormClient()
    return _client


def check_interactions_online(drug_names: List[str]) -> List[RxNormInteraction]:
    """
    Convenience function to check drug interactions via RxNorm API.

    Args:
        drug_names: List of drug names

    Returns:
        List of interactions
    """
    client = get_rxnorm_client()
    return client.check_interactions(drug_names)
