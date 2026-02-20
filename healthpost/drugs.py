"""Drug interaction checker using the DDInter API.

Uses the DDInter API for online drug-drug interaction checking (236,834+
interactions from DrugBank, KEGG, etc.).
"""

import logging
from typing import List, Set, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DrugInteraction(BaseModel):
    """A drug-drug interaction record."""

    drugs: Tuple[str, str]
    severity: str
    description: str
    recommendation: str


class DrugDatabase:
    """Drug interaction checker backed by the DDInter API.

    Provides online interaction checks via DDInter API (236K+ interactions).
    """

    def __init__(self) -> None:
        self._ddinter_client = None

    def _get_ddinter_client(self):
        """Return the DDInter API client, lazily initialized."""
        if self._ddinter_client is None:
            try:
                from .ddinter_api import DDInterClient
                self._ddinter_client = DDInterClient()
            except Exception as e:
                logger.warning(
                    "Failed to initialize DDInter client: %s", e,
                )
                self._ddinter_client = False
        return self._ddinter_client if self._ddinter_client else None

    def check_interactions(
        self, medications: List[str],
    ) -> List[DrugInteraction]:
        """Check for interactions between medications via DDInter.

        Args:
            medications: Medication names to check (needs >= 2).

        Returns:
            Interactions sorted by severity (severe first).
        """
        if len(medications) < 2:
            return []

        client = self._get_ddinter_client()
        if not client:
            logger.warning("DDInter client unavailable, cannot check interactions")
            return []

        try:
            ddinter_results = client.check_interactions(medications)

            result: List[DrugInteraction] = []
            seen: Set[Tuple[str, str]] = set()
            severity_map = {
                "major": "severe",
                "severe": "severe",
                "moderate": "moderate",
                "minor": "mild",
                "unknown": "moderate",
            }

            for r in ddinter_results:
                severity = severity_map.get(
                    r.severity.lower(), "moderate",
                )
                pair_key = tuple(
                    sorted([r.drug1.lower(), r.drug2.lower()])
                )
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                result.append(DrugInteraction(
                    drugs=(r.drug1, r.drug2),
                    severity=severity,
                    description=r.description,
                    recommendation=f"Source: DDInter. {r.management}",
                ))

            severity_order = {
                "severe": 0, "moderate": 1, "mild": 2,
            }
            result.sort(
                key=lambda x: severity_order.get(x.severity, 3),
            )
            if result:
                logger.info(
                    "Found %d interactions via DDInter API",
                    len(result),
                )
            return result
        except Exception as e:
            logger.warning("DDInter API check failed: %s", e)
            return []
