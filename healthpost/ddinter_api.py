"""DDInter API client for drug-drug interaction checking.

DDInter is a comprehensive drug-drug interaction database maintained by
SCBDD (South China Bioinformatics Data Laboratory). It contains 302,516
DDI associations between 2,290 drugs compiled from DrugBank, KEGG, and
other authoritative sources.

See https://ddinter2.scbdd.com/ for details.
"""

import json
import logging
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DDINTER_BASE = "https://ddinter2.scbdd.com"

DRUG_NAME_ALIASES: Dict[str, str] = {
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
    """A single drug-drug interaction returned by the DDInter API.

    Attributes:
        drug1: First drug name.
        drug2: Second drug name.
        severity: Severity level (``"Major"``, ``"Moderate"``,
            ``"Minor"``, or ``"Unknown"``).
        description: Clinical description of the interaction.
        management: Recommended management strategy.
    """

    drug1: str
    drug2: str
    severity: str
    description: str
    management: str


class DDInterClient:
    """HTTP client for the DDInter drug interaction checker.

    Attributes:
        base_url: DDInter server base URL.
    """

    def __init__(self) -> None:
        """Initialize the client with caches and rate-limit state."""
        self.base_url = DDINTER_BASE
        self._drug_cache: Dict[str, Optional[str]] = {}
        self._name_cache: Dict[str, str] = {}
        self._last_request_time = 0.0
        self._min_request_interval = 0.2
        self._ssl_context = ssl.create_default_context()

    def _rate_limit(self) -> None:
        """Sleep if needed to respect the minimum request interval."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _request_get(self, endpoint: str) -> Optional[str]:
        """Perform a rate-limited GET request.

        Args:
            endpoint: Path relative to *base_url*.

        Returns:
            Response body as a string, or ``None`` on failure.
        """
        self._rate_limit()
        url = f"{self.base_url}/{endpoint}"

        try:
            req = urllib.request.Request(url)
            req.add_header(
                "User-Agent",
                "HealthPost/1.0 (Medical Decision Support)",
            )
            req.add_header("Accept", "text/html,application/json")

            with urllib.request.urlopen(
                req, timeout=15, context=self._ssl_context,
            ) as response:
                return response.read().decode("utf-8")
        except urllib.error.URLError as e:
            logger.warning(
                "DDInter API request failed for %s: %s", endpoint, e,
            )
            return None
        except Exception as e:
            logger.error("DDInter API error: %s", e)
            return None

    def search_drug(self, drug_name: str) -> Optional[str]:
        """Look up a drug and return its DDInter internal ID.

        Args:
            drug_name: Drug name to search (minimum 3 characters).

        Returns:
            DDInter ID string (e.g. ``"DDInter1340"``), or ``None``.
        """
        cache_key = drug_name.lower().strip()
        if cache_key in self._drug_cache:
            return self._drug_cache[cache_key]

        search_term = DRUG_NAME_ALIASES.get(cache_key, drug_name.strip())
        # DDInter search endpoint requires at least 3 characters;
        # pad short names with spaces to satisfy the minimum length.
        if len(search_term) < 3:
            search_term = search_term + "   "

        endpoint = (
            f"check-datasource/"
            f"{urllib.parse.quote(search_term[:10])}/"
        )
        response = self._request_get(endpoint)

        if not response:
            self._drug_cache[cache_key] = None
            return None

        try:
            data = json.loads(response)
            results = data.get("data", [])

            if results:
                for result in results:
                    if result.get("name", "").lower() == cache_key:
                        ddinter_id = result.get("internalID")
                        self._drug_cache[cache_key] = ddinter_id
                        self._name_cache[ddinter_id] = result.get("name")
                        logger.debug(
                            "Found exact DDInter ID for %s: %s",
                            drug_name, ddinter_id,
                        )
                        return ddinter_id

                first_result = results[0]
                ddinter_id = first_result.get("internalID")
                self._drug_cache[cache_key] = ddinter_id
                self._name_cache[ddinter_id] = first_result.get("name")
                logger.debug(
                    "Found DDInter ID for %s: %s (%s)",
                    drug_name, ddinter_id, first_result.get("name"),
                )
                return ddinter_id

        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse DDInter search response for %s",
                drug_name,
            )

        self._drug_cache[cache_key] = None
        logger.debug("Drug not found in DDInter: %s", drug_name)
        return None

    def check_interactions(
        self, drug_names: List[str],
    ) -> List[DDInterInteraction]:
        """Check pairwise interactions between two to five drugs.

        Args:
            drug_names: Drug names to check (2-5 items).

        Returns:
            List of interactions found.
        """
        if len(drug_names) < 2:
            return []

        ddinter_ids: List[str] = []
        id_to_name: Dict[str, str] = {}

        for name in drug_names:
            ddinter_id = self.search_drug(name)
            if ddinter_id:
                ddinter_ids.append(ddinter_id)
                id_to_name[ddinter_id] = self._name_cache.get(
                    ddinter_id, name,
                )
            else:
                logger.warning(
                    "Could not find DDInter ID for: %s", name,
                )

        if len(ddinter_ids) < 2:
            logger.warning(
                "Not enough drugs found in DDInter to check interactions",
            )
            return []

        if len(ddinter_ids) > 5:
            logger.warning(
                "DDInter checker supports max 5 drugs, truncating list",
            )
            ddinter_ids = ddinter_ids[:5]

        ids_str = "-".join(ddinter_ids)
        endpoint = f"checker/result/{ids_str}/"
        html = self._request_get(endpoint)

        if not html:
            return []

        return self._parse_checker_results(html, id_to_name)

    def _parse_checker_results(
        self, html: str, id_to_name: Dict[str, str],
    ) -> List[DDInterInteraction]:
        """Extract interactions from the checker result HTML page.

        Args:
            html: Raw HTML from the checker endpoint.
            id_to_name: Mapping of DDInter IDs to display names.

        Returns:
            Parsed interaction list.
        """
        interactions: List[DDInterInteraction] = []

        pattern = r"let\s+response_data\s*=\s*(\[.*?\]);"
        match = re.search(pattern, html, re.DOTALL)

        if not match:
            logger.warning(
                "Could not find response_data in DDInter result page",
            )
            return []

        try:
            json_str = match.group(1).replace("'", '"')
            data = json.loads(json_str)

            for item in data:
                drug1 = item.get("drug_a_name", "Unknown")
                drug2 = item.get("drug_b_name", "Unknown")
                severity = item.get("idx__level", "Unknown")
                description = item.get(
                    "idx__interaction_description", "-",
                )
                management = item.get("idx__management", "-")

                if description == "-" and severity == "Unknown":
                    continue

                interactions.append(DDInterInteraction(
                    drug1=drug1,
                    drug2=drug2,
                    severity=severity,
                    description=(
                        description
                        if description != "-"
                        else f"Potential interaction between {drug1} and {drug2}"
                    ),
                    management=(
                        management
                        if management != "-"
                        else "Consult prescriber for guidance."
                    ),
                ))
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse DDInter response_data: %s", e)
        except Exception as e:
            logger.error("Error parsing DDInter results: %s", e)

        return interactions


_client: Optional[DDInterClient] = None


def get_ddinter_client() -> DDInterClient:
    """Return (and lazily create) the module-level DDInter client."""
    global _client
    if _client is None:
        _client = DDInterClient()
    return _client


def check_interactions_online(
    drug_names: List[str],
) -> List[DDInterInteraction]:
    """Convenience wrapper to check interactions via the DDInter API.

    Args:
        drug_names: Drug names to check.

    Returns:
        List of interactions found.
    """
    client = get_ddinter_client()
    return client.check_interactions(drug_names)
