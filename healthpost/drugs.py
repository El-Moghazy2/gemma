"""
Drug database and interaction checker.

Uses DDInter API for online drug interaction checking.
DDInter contains 236,834 drug-drug interactions from DrugBank, KEGG, etc.

Local SQLite database provides drug info (WHO Essential Medicines List).
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple
from pathlib import Path
import sqlite3
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class DrugInteraction:
    """Represents a drug-drug interaction."""
    drugs: Tuple[str, str]  # The two drugs involved
    severity: str  # "mild", "moderate", "severe"
    description: str  # Clinical description of the interaction
    recommendation: str  # What to do about it


@dataclass
class DrugInfo:
    """Information about a drug."""
    name: str
    generic_name: str
    drug_class: str
    common_uses: List[str]
    contraindications: List[str]
    common_doses: Dict[str, str]  # indication -> dose


class DrugDatabase:
    """
    Drug database with online interaction checking via DDInter API.

    Features:
    - Online: DDInter API for comprehensive drug interactions (236K+ interactions)
    - Local SQLite for drug info (WHO Essential Medicines)
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the drug database.

        Args:
            db_path: Path to SQLite database for drug info
        """
        self.db_path = db_path or Path(__file__).parent.parent / "data" / "drugs.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._ddinter_client = None

        # Ensure database exists for drug info
        self._init_database()

    def _get_ddinter_client(self):
        """Get DDInter API client (lazy loaded)."""
        if self._ddinter_client is None:
            try:
                from .ddinter_api import DDInterClient
                self._ddinter_client = DDInterClient()
            except Exception as e:
                logger.warning(f"Failed to initialize DDInter client: {e}")
                self._ddinter_client = False  # Mark as unavailable
        return self._ddinter_client if self._ddinter_client else None

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_database(self):
        """Initialize the database with schema and essential data."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = self._get_connection()
        cursor = conn.cursor()

        # Create tables
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS drugs (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                generic_name TEXT,
                drug_class TEXT,
                common_uses TEXT,  -- JSON array
                contraindications TEXT,  -- JSON array
                common_doses TEXT  -- JSON object
            );

            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY,
                drug1 TEXT NOT NULL,
                drug2 TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                recommendation TEXT,
                UNIQUE(drug1, drug2)
            );

            CREATE INDEX IF NOT EXISTS idx_interactions_drug1 ON interactions(drug1);
            CREATE INDEX IF NOT EXISTS idx_interactions_drug2 ON interactions(drug2);
            CREATE INDEX IF NOT EXISTS idx_drugs_name ON drugs(name);
            CREATE INDEX IF NOT EXISTS idx_drugs_generic ON drugs(generic_name);
        """)

        conn.commit()

        # Populate with essential medicines data if empty
        cursor.execute("SELECT COUNT(*) FROM drugs")
        if cursor.fetchone()[0] == 0:
            self._populate_essential_medicines()

    def _populate_essential_medicines(self):
        """Populate database with WHO Essential Medicines data."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Essential medicines commonly used by CHWs
        essential_drugs = [
            # Analgesics / Antipyretics
            ("Paracetamol", "Acetaminophen", "Analgesic",
             ["fever", "pain", "headache"],
             ["liver disease", "alcohol use disorder"],
             {"fever": "500-1000mg every 4-6 hours, max 4g/day",
              "pain": "500-1000mg every 4-6 hours"}),

            ("Ibuprofen", "Ibuprofen", "NSAID",
             ["pain", "inflammation", "fever"],
             ["peptic ulcer", "kidney disease", "aspirin allergy", "pregnancy third trimester"],
             {"pain": "200-400mg every 4-6 hours", "fever": "200-400mg every 4-6 hours"}),

            ("Aspirin", "Acetylsalicylic acid", "NSAID",
             ["pain", "fever", "heart protection"],
             ["peptic ulcer", "bleeding disorders", "children under 16 with viral illness"],
             {"pain": "300-600mg every 4-6 hours", "heart": "75-100mg daily"}),

            # Antibiotics
            ("Amoxicillin", "Amoxicillin", "Antibiotic-Penicillin",
             ["respiratory infection", "ear infection", "urinary infection", "skin infection"],
             ["penicillin allergy"],
             {"adult": "500mg every 8 hours", "child": "25mg/kg/day divided every 8 hours"}),

            ("Metronidazole", "Metronidazole", "Antibiotic-Nitroimidazole",
             ["giardiasis", "amoebiasis", "bacterial vaginosis", "dental infection"],
             ["first trimester pregnancy", "alcohol use"],
             {"adult": "400mg every 8 hours", "giardiasis": "2g single dose"}),

            ("Cotrimoxazole", "Sulfamethoxazole-Trimethoprim", "Antibiotic-Sulfonamide",
             ["urinary infection", "respiratory infection", "diarrhea"],
             ["sulfa allergy", "severe kidney disease", "first trimester pregnancy"],
             {"adult": "960mg every 12 hours"}),

            ("Doxycycline", "Doxycycline", "Antibiotic-Tetracycline",
             ["malaria prophylaxis", "respiratory infection", "skin infection", "STI"],
             ["pregnancy", "children under 8", "severe liver disease"],
             {"adult": "100mg every 12 hours"}),

            ("Ciprofloxacin", "Ciprofloxacin", "Antibiotic-Fluoroquinolone",
             ["urinary infection", "diarrhea", "respiratory infection"],
             ["pregnancy", "children", "tendon disorders"],
             {"adult": "500mg every 12 hours"}),

            # Antimalarials
            ("Artemether-Lumefantrine", "Artemether-Lumefantrine", "Antimalarial",
             ["uncomplicated malaria"],
             ["first trimester pregnancy", "severe malaria"],
             {"adult": "4 tablets at 0, 8, 24, 36, 48, 60 hours"}),

            ("Chloroquine", "Chloroquine", "Antimalarial",
             ["malaria treatment", "malaria prophylaxis"],
             ["retinal disease", "psoriasis"],
             {"prophylaxis": "500mg weekly", "treatment": "10mg/kg day1, 5mg/kg day2-3"}),

            # Antihypertensives
            ("Amlodipine", "Amlodipine", "Calcium channel blocker",
             ["hypertension", "angina"],
             ["severe aortic stenosis", "unstable angina"],
             {"hypertension": "5-10mg once daily"}),

            ("Enalapril", "Enalapril", "ACE inhibitor",
             ["hypertension", "heart failure"],
             ["pregnancy", "angioedema history", "bilateral renal artery stenosis"],
             {"hypertension": "5-20mg once or twice daily"}),

            ("Hydrochlorothiazide", "Hydrochlorothiazide", "Diuretic-Thiazide",
             ["hypertension", "edema"],
             ["severe kidney disease", "gout"],
             {"hypertension": "12.5-25mg daily"}),

            ("Metoprolol", "Metoprolol", "Beta blocker",
             ["hypertension", "angina", "heart failure"],
             ["severe asthma", "severe bradycardia", "heart block"],
             {"hypertension": "50-100mg twice daily"}),

            # Diabetes medications
            ("Metformin", "Metformin", "Antidiabetic-Biguanide",
             ["type 2 diabetes"],
             ["kidney disease", "liver disease", "heart failure", "alcohol use disorder"],
             {"diabetes": "500mg twice daily, increase gradually to 1000mg twice daily"}),

            ("Glibenclamide", "Glyburide", "Antidiabetic-Sulfonylurea",
             ["type 2 diabetes"],
             ["type 1 diabetes", "severe kidney disease", "pregnancy"],
             {"diabetes": "2.5-5mg daily"}),

            # Respiratory
            ("Salbutamol", "Albuterol", "Bronchodilator",
             ["asthma", "COPD", "bronchospasm"],
             ["uncontrolled hyperthyroidism"],
             {"asthma": "2 puffs every 4-6 hours as needed"}),

            # Gastrointestinal
            ("Omeprazole", "Omeprazole", "Proton pump inhibitor",
             ["peptic ulcer", "GERD", "gastritis"],
             [],
             {"ulcer": "20-40mg daily"}),

            ("Oral Rehydration Salts", "ORS", "Electrolyte replacement",
             ["diarrhea", "dehydration"],
             [],
             {"dehydration": "75ml/kg over 4 hours for moderate dehydration"}),

            ("Loperamide", "Loperamide", "Antidiarrheal",
             ["acute diarrhea"],
             ["bloody diarrhea", "fever with diarrhea", "children under 2"],
             {"adult": "4mg initially, then 2mg after each loose stool, max 16mg/day"}),

            # Vitamins and supplements
            ("Vitamin A", "Retinol", "Vitamin",
             ["vitamin A deficiency", "measles"],
             ["pregnancy in high doses"],
             {"measles": "200,000 IU single dose for children over 1 year"}),

            ("Iron-Folic Acid", "Ferrous sulfate with folic acid", "Supplement",
             ["anemia", "pregnancy supplementation"],
             ["iron overload", "hemochromatosis"],
             {"anemia": "1 tablet daily"}),

            ("Zinc", "Zinc sulfate", "Supplement",
             ["diarrhea in children", "zinc deficiency"],
             [],
             {"diarrhea": "20mg daily for 10-14 days for children over 6 months"}),

            # Antihistamines
            ("Chlorpheniramine", "Chlorpheniramine", "Antihistamine",
             ["allergic reactions", "itching", "rhinitis"],
             ["severe asthma attack"],
             {"allergy": "4mg every 4-6 hours"}),

            ("Promethazine", "Promethazine", "Antihistamine-Sedating",
             ["allergic reactions", "nausea", "sedation"],
             ["children under 2", "respiratory depression"],
             {"allergy": "25mg at night or twice daily"}),

            # Antifungals
            ("Clotrimazole", "Clotrimazole", "Antifungal",
             ["fungal skin infection", "vaginal candidiasis"],
             [],
             {"skin": "Apply twice daily for 2-4 weeks", "vaginal": "500mg single dose pessary"}),

            # Pain - stronger
            ("Tramadol", "Tramadol", "Opioid analgesic",
             ["moderate to severe pain"],
             ["seizure disorders", "MAO inhibitors", "respiratory depression"],
             {"pain": "50-100mg every 4-6 hours, max 400mg/day"}),

            ("Codeine", "Codeine", "Opioid analgesic",
             ["moderate pain", "cough"],
             ["respiratory depression", "children under 12"],
             {"pain": "30-60mg every 4-6 hours"}),

            # Corticosteroids
            ("Prednisolone", "Prednisolone", "Corticosteroid",
             ["inflammation", "allergic reactions", "asthma exacerbation"],
             ["systemic fungal infection", "live vaccines"],
             {"asthma": "40-60mg daily for 5-7 days"}),

            ("Hydrocortisone cream", "Hydrocortisone", "Topical corticosteroid",
             ["eczema", "dermatitis", "insect bites"],
             ["skin infections", "rosacea"],
             {"dermatitis": "Apply thin layer 1-2 times daily"}),

            # Anticonvulsants
            ("Phenobarbital", "Phenobarbital", "Anticonvulsant",
             ["epilepsy", "seizures"],
             ["porphyria", "severe respiratory disease"],
             {"epilepsy": "60-180mg at night"}),

            # Antipsychotics
            ("Haloperidol", "Haloperidol", "Antipsychotic",
             ["psychosis", "severe agitation"],
             ["Parkinson's disease", "severe cardiac disease"],
             {"acute agitation": "2-5mg IM"}),
        ]

        for drug in essential_drugs:
            cursor.execute("""
                INSERT OR IGNORE INTO drugs
                (name, generic_name, drug_class, common_uses, contraindications, common_doses)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                drug[0],
                drug[1],
                drug[2],
                json.dumps(drug[3]),
                json.dumps(drug[4]),
                json.dumps(drug[5]),
            ))

        # Important drug interactions
        interactions = [
            # Severe interactions
            ("Metformin", "Alcohol", "severe",
             "Increased risk of lactic acidosis",
             "Avoid alcohol with metformin"),

            ("Warfarin", "Aspirin", "severe",
             "Greatly increased bleeding risk",
             "Avoid combination unless specifically indicated"),

            ("Methotrexate", "Cotrimoxazole", "severe",
             "Increased methotrexate toxicity, bone marrow suppression",
             "Avoid combination"),

            ("MAO inhibitors", "Tramadol", "severe",
             "Risk of serotonin syndrome",
             "Contraindicated - do not use together"),

            ("Metronidazole", "Alcohol", "severe",
             "Disulfiram-like reaction: severe nausea, vomiting, flushing",
             "No alcohol during treatment and 48 hours after"),

            ("Ciprofloxacin", "Theophylline", "severe",
             "Increased theophylline levels, risk of toxicity",
             "Monitor theophylline levels, consider dose reduction"),

            ("ACE inhibitors", "Potassium supplements", "severe",
             "Risk of dangerous hyperkalemia",
             "Monitor potassium levels closely"),

            # Moderate interactions
            ("NSAIDs", "ACE inhibitors", "moderate",
             "Reduced antihypertensive effect, increased kidney risk",
             "Use lowest NSAID dose for shortest time"),

            ("NSAIDs", "Aspirin", "moderate",
             "Increased GI bleeding risk",
             "Consider gastroprotection if combination necessary"),

            ("Metformin", "Contrast dye", "moderate",
             "Risk of lactic acidosis",
             "Hold metformin before and 48 hours after contrast"),

            ("Ciprofloxacin", "Antacids", "moderate",
             "Reduced ciprofloxacin absorption",
             "Take ciprofloxacin 2 hours before or 6 hours after antacids"),

            ("Amlodipine", "Grapefruit", "moderate",
             "Increased amlodipine levels",
             "Avoid grapefruit juice"),

            ("Doxycycline", "Antacids", "moderate",
             "Reduced doxycycline absorption",
             "Separate administration by 2-3 hours"),

            ("Codeine", "Promethazine", "moderate",
             "Increased sedation and respiratory depression",
             "Use lower doses, monitor closely"),

            ("Beta blockers", "Salbutamol", "moderate",
             "Reduced bronchodilator effect",
             "May need higher salbutamol doses"),

            # Mild interactions
            ("Paracetamol", "Alcohol", "mild",
             "Increased risk of liver damage with chronic alcohol use",
             "Limit alcohol intake"),

            ("Omeprazole", "Iron supplements", "mild",
             "Reduced iron absorption",
             "Consider taking iron with vitamin C"),

            ("Metformin", "Vitamin B12", "mild",
             "Long-term use may reduce B12 absorption",
             "Monitor B12 levels yearly"),
        ]

        for interaction in interactions:
            # Insert both directions for easier lookup
            cursor.execute("""
                INSERT OR IGNORE INTO interactions
                (drug1, drug2, severity, description, recommendation)
                VALUES (?, ?, ?, ?, ?)
            """, interaction)

            cursor.execute("""
                INSERT OR IGNORE INTO interactions
                (drug1, drug2, severity, description, recommendation)
                VALUES (?, ?, ?, ?, ?)
            """, (interaction[1], interaction[0], interaction[2], interaction[3], interaction[4]))

        conn.commit()
        logger.info(f"Populated database with {len(essential_drugs)} drugs and {len(interactions)} interactions")

    def check_interactions(self, medications: List[str]) -> List[DrugInteraction]:
        """
        Check for interactions between a list of medications using DDInter API.

        Args:
            medications: List of medication names to check

        Returns:
            List of DrugInteraction objects for any interactions found
        """
        if len(medications) < 2:
            return []

        client = self._get_ddinter_client()
        if not client:
            logger.error("DDInter client unavailable - cannot check interactions")
            return []

        try:
            from .ddinter_api import DDInterInteraction
            ddinter_results = client.check_interactions(medications)

            # Convert to DrugInteraction format
            interactions = []
            seen = set()

            for r in ddinter_results:
                # Normalize severity
                severity_map = {
                    "major": "severe",
                    "severe": "severe",
                    "moderate": "moderate",
                    "minor": "mild",
                    "unknown": "moderate",  # Default to moderate if unknown
                }
                severity = severity_map.get(r.severity.lower(), "moderate")

                # Avoid duplicates
                pair_key = tuple(sorted([r.drug1.lower(), r.drug2.lower()]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                interactions.append(DrugInteraction(
                    drugs=(r.drug1, r.drug2),
                    severity=severity,
                    description=r.description,
                    recommendation=f"Source: DDInter. {r.management}",
                ))

            # Sort by severity
            severity_order = {"severe": 0, "moderate": 1, "mild": 2}
            interactions.sort(key=lambda x: severity_order.get(x.severity, 3))

            logger.info(f"Found {len(interactions)} interactions via DDInter API")
            return interactions

        except Exception as e:
            logger.error(f"DDInter API check failed: {e}")
            return []

    def _check_interactions_local(self, medications: List[str]) -> List[DrugInteraction]:
        """Check interactions using local database."""
        interactions = []
        conn = self._get_connection()
        cursor = conn.cursor()

        # Normalize medication names
        meds_normalized = [self._normalize_drug_name(m) for m in medications]

        # Check each pair
        checked_pairs: Set[Tuple[str, str]] = set()

        for i, med1 in enumerate(meds_normalized):
            for med2 in meds_normalized[i+1:]:
                # Skip if already checked
                pair = tuple(sorted([med1, med2]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                # Check for direct interaction
                interaction = self._find_interaction(cursor, med1, med2)
                if interaction:
                    interactions.append(interaction)

                # Check by drug class
                class_interaction = self._find_class_interaction(cursor, med1, med2)
                if class_interaction and class_interaction not in interactions:
                    interactions.append(class_interaction)

        # Sort by severity
        severity_order = {"severe": 0, "moderate": 1, "mild": 2}
        interactions.sort(key=lambda x: severity_order.get(x.severity, 3))

        return interactions

    def _normalize_drug_name(self, name: str) -> str:
        """Normalize drug name for lookup."""
        # Remove common suffixes and standardize
        name = name.lower().strip()

        # Remove dosage info
        import re
        name = re.sub(r'\d+\s*mg', '', name)
        name = re.sub(r'\d+\s*ml', '', name)
        name = re.sub(r'\d+%', '', name)

        # Remove common suffixes
        suffixes = ['tablet', 'tablets', 'capsule', 'capsules', 'cream', 'ointment',
                   'syrup', 'injection', 'solution', 'suspension']
        for suffix in suffixes:
            name = name.replace(suffix, '')

        return name.strip()

    def _find_interaction(
        self, cursor: sqlite3.Cursor, drug1: str, drug2: str
    ) -> Optional[DrugInteraction]:
        """Find direct interaction between two drugs."""

        # Try exact match first
        cursor.execute("""
            SELECT severity, description, recommendation
            FROM interactions
            WHERE LOWER(drug1) = ? AND LOWER(drug2) = ?
        """, (drug1, drug2))

        row = cursor.fetchone()
        if row:
            return DrugInteraction(
                drugs=(drug1, drug2),
                severity=row["severity"],
                description=row["description"],
                recommendation=row["recommendation"],
            )

        # Try partial match
        cursor.execute("""
            SELECT drug1, drug2, severity, description, recommendation
            FROM interactions
            WHERE LOWER(drug1) LIKE ? AND LOWER(drug2) LIKE ?
        """, (f"%{drug1}%", f"%{drug2}%"))

        row = cursor.fetchone()
        if row:
            return DrugInteraction(
                drugs=(row["drug1"], row["drug2"]),
                severity=row["severity"],
                description=row["description"],
                recommendation=row["recommendation"],
            )

        return None

    def _find_class_interaction(
        self, cursor: sqlite3.Cursor, drug1: str, drug2: str
    ) -> Optional[DrugInteraction]:
        """Find interaction based on drug classes."""

        # Get drug classes
        cursor.execute("""
            SELECT name, drug_class FROM drugs
            WHERE LOWER(name) LIKE ? OR LOWER(generic_name) LIKE ?
        """, (f"%{drug1}%", f"%{drug1}%"))
        row1 = cursor.fetchone()

        cursor.execute("""
            SELECT name, drug_class FROM drugs
            WHERE LOWER(name) LIKE ? OR LOWER(generic_name) LIKE ?
        """, (f"%{drug2}%", f"%{drug2}%"))
        row2 = cursor.fetchone()

        if not row1 or not row2:
            return None

        class1 = row1["drug_class"] if row1 else None
        class2 = row2["drug_class"] if row2 else None

        if not class1 or not class2:
            return None

        # Check for class-based interactions
        cursor.execute("""
            SELECT severity, description, recommendation
            FROM interactions
            WHERE LOWER(drug1) LIKE ? AND LOWER(drug2) LIKE ?
        """, (f"%{class1.lower()}%", f"%{class2.lower()}%"))

        row = cursor.fetchone()
        if row:
            return DrugInteraction(
                drugs=(drug1, drug2),
                severity=row["severity"],
                description=f"{row['description']} (class interaction: {class1} + {class2})",
                recommendation=row["recommendation"],
            )

        return None

    def get_drug_info(self, drug_name: str) -> Optional[DrugInfo]:
        """Get information about a specific drug."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM drugs
            WHERE LOWER(name) LIKE ? OR LOWER(generic_name) LIKE ?
        """, (f"%{drug_name.lower()}%", f"%{drug_name.lower()}%"))

        row = cursor.fetchone()
        if not row:
            return None

        return DrugInfo(
            name=row["name"],
            generic_name=row["generic_name"],
            drug_class=row["drug_class"],
            common_uses=json.loads(row["common_uses"]) if row["common_uses"] else [],
            contraindications=json.loads(row["contraindications"]) if row["contraindications"] else [],
            common_doses=json.loads(row["common_doses"]) if row["common_doses"] else {},
        )

    def search_drugs(self, query: str, limit: int = 10) -> List[DrugInfo]:
        """Search for drugs matching a query."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM drugs
            WHERE LOWER(name) LIKE ? OR LOWER(generic_name) LIKE ?
            OR LOWER(drug_class) LIKE ?
            LIMIT ?
        """, (f"%{query.lower()}%", f"%{query.lower()}%", f"%{query.lower()}%", limit))

        results = []
        for row in cursor.fetchall():
            results.append(DrugInfo(
                name=row["name"],
                generic_name=row["generic_name"],
                drug_class=row["drug_class"],
                common_uses=json.loads(row["common_uses"]) if row["common_uses"] else [],
                contraindications=json.loads(row["contraindications"]) if row["contraindications"] else [],
                common_doses=json.loads(row["common_doses"]) if row["common_doses"] else {},
            ))

        return results

    def check_contraindications(
        self, drug_name: str, conditions: List[str]
    ) -> List[str]:
        """
        Check if a drug is contraindicated for any of the given conditions.

        Returns list of matching contraindications.
        """
        drug_info = self.get_drug_info(drug_name)
        if not drug_info:
            return []

        matches = []
        for condition in conditions:
            condition_lower = condition.lower()
            for contra in drug_info.contraindications:
                if condition_lower in contra.lower() or contra.lower() in condition_lower:
                    matches.append(contra)

        return matches

    def get_dosage(self, drug_name: str, indication: Optional[str] = None) -> Optional[str]:
        """Get dosage information for a drug."""
        drug_info = self.get_drug_info(drug_name)
        if not drug_info or not drug_info.common_doses:
            return None

        if indication:
            # Try to find matching indication
            indication_lower = indication.lower()
            for ind, dose in drug_info.common_doses.items():
                if indication_lower in ind.lower() or ind.lower() in indication_lower:
                    return dose

        # Return first available dose
        return next(iter(drug_info.common_doses.values()), None)

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
