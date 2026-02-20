"""Drug database and interaction checker.

Uses the DDInter API for online drug-drug interaction checking (236,834+
interactions from DrugBank, KEGG, etc.) and a local SQLite database for
drug information based on the WHO Essential Medicines List.
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DrugInteraction:
    """A drug-drug interaction record.

    Attributes:
        drugs: The two drugs involved.
        severity: One of ``"mild"``, ``"moderate"``, ``"severe"``.
        description: Clinical description of the interaction.
        recommendation: Recommended action.
    """

    drugs: Tuple[str, str]
    severity: str
    description: str
    recommendation: str


@dataclass
class DrugInfo:
    """Reference information for a single drug.

    Attributes:
        name: Trade or common name.
        generic_name: International nonproprietary name.
        drug_class: Pharmacological class.
        common_uses: Typical indications.
        contraindications: Known contraindications.
        common_doses: Mapping of indication to dosage string.
    """

    name: str
    generic_name: str
    drug_class: str
    common_uses: List[str]
    contraindications: List[str]
    common_doses: Dict[str, str]


class DrugDatabase:
    """SQLite-backed drug database with online interaction checking.

    Provides:
    - Online interaction checks via DDInter API (236K+ interactions).
    - Local SQLite storage for drug reference data (WHO Essential
      Medicines).
    - Class-based interaction detection when direct matches are absent.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize the drug database.

        Args:
            db_path: Path to the SQLite database.  Defaults to
                ``data/drugs.db`` relative to the package root.
        """
        self.db_path = (
            db_path
            or Path(__file__).parent.parent / "data" / "drugs.db"
        )
        self._conn: Optional[sqlite3.Connection] = None
        self._ddinter_client = None
        self._init_database()

    def _get_ddinter_client(self):
        """Return the DDInter API client, lazily initialized.

        Returns:
            ``DDInterClient`` instance, or ``None`` if unavailable.
        """
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

    def _get_connection(self) -> sqlite3.Connection:
        """Return the (lazily opened) database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_database(self) -> None:
        """Create schema and seed essential medicines data if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS drugs (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                generic_name TEXT,
                drug_class TEXT,
                common_uses TEXT,
                contraindications TEXT,
                common_doses TEXT
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

            CREATE INDEX IF NOT EXISTS idx_interactions_drug1
                ON interactions(drug1);
            CREATE INDEX IF NOT EXISTS idx_interactions_drug2
                ON interactions(drug2);
            CREATE INDEX IF NOT EXISTS idx_drugs_name
                ON drugs(name);
            CREATE INDEX IF NOT EXISTS idx_drugs_generic
                ON drugs(generic_name);
        """)
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM drugs")
        if cursor.fetchone()[0] == 0:
            self._populate_essential_medicines()

    def _populate_essential_medicines(self) -> None:
        """Seed the database with WHO Essential Medicines data."""
        conn = self._get_connection()
        cursor = conn.cursor()

        essential_drugs = [
            ("Paracetamol", "Acetaminophen", "Analgesic",
             ["fever", "pain", "headache"],
             ["liver disease", "alcohol use disorder"],
             {"fever": "500-1000mg every 4-6 hours, max 4g/day",
              "pain": "500-1000mg every 4-6 hours"}),
            ("Ibuprofen", "Ibuprofen", "NSAID",
             ["pain", "inflammation", "fever"],
             ["peptic ulcer", "kidney disease", "aspirin allergy",
              "pregnancy third trimester"],
             {"pain": "200-400mg every 4-6 hours",
              "fever": "200-400mg every 4-6 hours"}),
            ("Aspirin", "Acetylsalicylic acid", "NSAID",
             ["pain", "fever", "heart protection"],
             ["peptic ulcer", "bleeding disorders",
              "children under 16 with viral illness"],
             {"pain": "300-600mg every 4-6 hours",
              "heart": "75-100mg daily"}),
            ("Amoxicillin", "Amoxicillin", "Antibiotic-Penicillin",
             ["respiratory infection", "ear infection",
              "urinary infection", "skin infection"],
             ["penicillin allergy"],
             {"adult": "500mg every 8 hours",
              "child": "25mg/kg/day divided every 8 hours"}),
            ("Metronidazole", "Metronidazole",
             "Antibiotic-Nitroimidazole",
             ["giardiasis", "amoebiasis", "bacterial vaginosis",
              "dental infection"],
             ["first trimester pregnancy", "alcohol use"],
             {"adult": "400mg every 8 hours",
              "giardiasis": "2g single dose"}),
            ("Cotrimoxazole", "Sulfamethoxazole-Trimethoprim",
             "Antibiotic-Sulfonamide",
             ["urinary infection", "respiratory infection", "diarrhea"],
             ["sulfa allergy", "severe kidney disease",
              "first trimester pregnancy"],
             {"adult": "960mg every 12 hours"}),
            ("Doxycycline", "Doxycycline", "Antibiotic-Tetracycline",
             ["malaria prophylaxis", "respiratory infection",
              "skin infection", "STI"],
             ["pregnancy", "children under 8",
              "severe liver disease"],
             {"adult": "100mg every 12 hours"}),
            ("Ciprofloxacin", "Ciprofloxacin",
             "Antibiotic-Fluoroquinolone",
             ["urinary infection", "diarrhea",
              "respiratory infection"],
             ["pregnancy", "children", "tendon disorders"],
             {"adult": "500mg every 12 hours"}),
            ("Artemether-Lumefantrine", "Artemether-Lumefantrine",
             "Antimalarial",
             ["uncomplicated malaria"],
             ["first trimester pregnancy", "severe malaria"],
             {"adult": "4 tablets at 0, 8, 24, 36, 48, 60 hours"}),
            ("Chloroquine", "Chloroquine", "Antimalarial",
             ["malaria treatment", "malaria prophylaxis"],
             ["retinal disease", "psoriasis"],
             {"prophylaxis": "500mg weekly",
              "treatment": "10mg/kg day1, 5mg/kg day2-3"}),
            ("Amlodipine", "Amlodipine", "Calcium channel blocker",
             ["hypertension", "angina"],
             ["severe aortic stenosis", "unstable angina"],
             {"hypertension": "5-10mg once daily"}),
            ("Enalapril", "Enalapril", "ACE inhibitor",
             ["hypertension", "heart failure"],
             ["pregnancy", "angioedema history",
              "bilateral renal artery stenosis"],
             {"hypertension": "5-20mg once or twice daily"}),
            ("Hydrochlorothiazide", "Hydrochlorothiazide",
             "Diuretic-Thiazide",
             ["hypertension", "edema"],
             ["severe kidney disease", "gout"],
             {"hypertension": "12.5-25mg daily"}),
            ("Metoprolol", "Metoprolol", "Beta blocker",
             ["hypertension", "angina", "heart failure"],
             ["severe asthma", "severe bradycardia", "heart block"],
             {"hypertension": "50-100mg twice daily"}),
            ("Metformin", "Metformin", "Antidiabetic-Biguanide",
             ["type 2 diabetes"],
             ["kidney disease", "liver disease", "heart failure",
              "alcohol use disorder"],
             {"diabetes": "500mg twice daily, increase gradually "
              "to 1000mg twice daily"}),
            ("Glibenclamide", "Glyburide",
             "Antidiabetic-Sulfonylurea",
             ["type 2 diabetes"],
             ["type 1 diabetes", "severe kidney disease", "pregnancy"],
             {"diabetes": "2.5-5mg daily"}),
            ("Salbutamol", "Albuterol", "Bronchodilator",
             ["asthma", "COPD", "bronchospasm"],
             ["uncontrolled hyperthyroidism"],
             {"asthma": "2 puffs every 4-6 hours as needed"}),
            ("Omeprazole", "Omeprazole", "Proton pump inhibitor",
             ["peptic ulcer", "GERD", "gastritis"],
             [],
             {"ulcer": "20-40mg daily"}),
            ("Oral Rehydration Salts", "ORS",
             "Electrolyte replacement",
             ["diarrhea", "dehydration"],
             [],
             {"dehydration": "75ml/kg over 4 hours for moderate "
              "dehydration"}),
            ("Loperamide", "Loperamide", "Antidiarrheal",
             ["acute diarrhea"],
             ["bloody diarrhea", "fever with diarrhea",
              "children under 2"],
             {"adult": "4mg initially, then 2mg after each loose "
              "stool, max 16mg/day"}),
            ("Vitamin A", "Retinol", "Vitamin",
             ["vitamin A deficiency", "measles"],
             ["pregnancy in high doses"],
             {"measles": "200,000 IU single dose for children "
              "over 1 year"}),
            ("Iron-Folic Acid", "Ferrous sulfate with folic acid",
             "Supplement",
             ["anemia", "pregnancy supplementation"],
             ["iron overload", "hemochromatosis"],
             {"anemia": "1 tablet daily"}),
            ("Zinc", "Zinc sulfate", "Supplement",
             ["diarrhea in children", "zinc deficiency"],
             [],
             {"diarrhea": "20mg daily for 10-14 days for children "
              "over 6 months"}),
            ("Chlorpheniramine", "Chlorpheniramine", "Antihistamine",
             ["allergic reactions", "itching", "rhinitis"],
             ["severe asthma attack"],
             {"allergy": "4mg every 4-6 hours"}),
            ("Promethazine", "Promethazine",
             "Antihistamine-Sedating",
             ["allergic reactions", "nausea", "sedation"],
             ["children under 2", "respiratory depression"],
             {"allergy": "25mg at night or twice daily"}),
            ("Clotrimazole", "Clotrimazole", "Antifungal",
             ["fungal skin infection", "vaginal candidiasis"],
             [],
             {"skin": "Apply twice daily for 2-4 weeks",
              "vaginal": "500mg single dose pessary"}),
            ("Tramadol", "Tramadol", "Opioid analgesic",
             ["moderate to severe pain"],
             ["seizure disorders", "MAO inhibitors",
              "respiratory depression"],
             {"pain": "50-100mg every 4-6 hours, max 400mg/day"}),
            ("Codeine", "Codeine", "Opioid analgesic",
             ["moderate pain", "cough"],
             ["respiratory depression", "children under 12"],
             {"pain": "30-60mg every 4-6 hours"}),
            ("Prednisolone", "Prednisolone", "Corticosteroid",
             ["inflammation", "allergic reactions",
              "asthma exacerbation"],
             ["systemic fungal infection", "live vaccines"],
             {"asthma": "40-60mg daily for 5-7 days"}),
            ("Hydrocortisone cream", "Hydrocortisone",
             "Topical corticosteroid",
             ["eczema", "dermatitis", "insect bites"],
             ["skin infections", "rosacea"],
             {"dermatitis": "Apply thin layer 1-2 times daily"}),
            ("Phenobarbital", "Phenobarbital", "Anticonvulsant",
             ["epilepsy", "seizures"],
             ["porphyria", "severe respiratory disease"],
             {"epilepsy": "60-180mg at night"}),
            ("Haloperidol", "Haloperidol", "Antipsychotic",
             ["psychosis", "severe agitation"],
             ["Parkinson's disease", "severe cardiac disease"],
             {"acute agitation": "2-5mg IM"}),
            ("Azithromycin", "Azithromycin", "Antibiotic-Macrolide",
             ["respiratory infection", "STI", "trachoma", "typhoid"],
             ["severe liver disease", "QT prolongation"],
             {"adult": "500mg day 1, then 250mg days 2-5",
              "STI": "1g single dose"}),
            ("Ceftriaxone", "Ceftriaxone",
             "Antibiotic-Cephalosporin",
             ["meningitis", "severe pneumonia", "gonorrhea", "sepsis"],
             ["cephalosporin allergy"],
             {"adult": "1-2g IV/IM daily",
              "gonorrhea": "250mg IM single dose"}),
            ("Gentamicin", "Gentamicin", "Antibiotic-Aminoglycoside",
             ["severe infection", "sepsis", "UTI"],
             ["kidney disease", "myasthenia gravis", "pregnancy"],
             {"adult": "5-7mg/kg IV daily"}),
            ("Erythromycin", "Erythromycin", "Antibiotic-Macrolide",
             ["respiratory infection", "skin infection", "chlamydia"],
             ["liver disease", "QT prolongation"],
             {"adult": "250-500mg every 6 hours"}),
            ("Fluconazole", "Fluconazole", "Antifungal-Azole",
             ["vaginal candidiasis", "oral thrush",
              "cryptococcal meningitis"],
             ["pregnancy", "severe liver disease"],
             {"vaginal": "150mg single dose",
              "oral thrush": "100mg daily for 7-14 days"}),
            ("Albendazole", "Albendazole", "Antihelminthic",
             ["roundworm", "hookworm", "whipworm",
              "hydatid disease"],
             ["first trimester pregnancy"],
             {"deworming": "400mg single dose",
              "hydatid": "400mg twice daily for 28 days"}),
            ("Mebendazole", "Mebendazole", "Antihelminthic",
             ["roundworm", "hookworm", "whipworm", "pinworm"],
             ["first trimester pregnancy"],
             {"deworming": "500mg single dose or 100mg twice daily "
              "for 3 days"}),
            ("Praziquantel", "Praziquantel", "Antihelminthic",
             ["schistosomiasis", "tapeworm"],
             ["ocular cysticercosis"],
             {"schistosomiasis": "40mg/kg single dose"}),
            ("Diazepam", "Diazepam", "Benzodiazepine",
             ["seizures", "anxiety", "muscle spasm",
              "alcohol withdrawal"],
             ["severe respiratory depression", "sleep apnea",
              "myasthenia gravis"],
             {"seizure": "5-10mg IV/rectal",
              "anxiety": "2-5mg 2-3 times daily"}),
            ("Oxytocin", "Oxytocin", "Uterotonic",
             ["postpartum hemorrhage", "labor induction"],
             ["hypertonic uterus"],
             {"PPH": "10 IU IM after delivery of placenta"}),
            ("Misoprostol", "Misoprostol", "Prostaglandin",
             ["postpartum hemorrhage", "gastric ulcer"],
             ["pregnancy (for ulcer use)"],
             {"PPH": "600mcg sublingual",
              "ulcer": "200mcg 4 times daily"}),
            ("Ferrous Sulfate", "Ferrous sulfate", "Iron supplement",
             ["iron deficiency anemia"],
             ["hemochromatosis", "hemosiderosis"],
             {"anemia": "200mg (65mg elemental iron) 2-3 times "
              "daily"}),
            ("Folic Acid", "Folic acid", "Vitamin B9",
             ["folate deficiency", "pregnancy supplementation",
              "megaloblastic anemia"],
             [],
             {"pregnancy": "400mcg daily",
              "deficiency": "5mg daily"}),
            ("Amoxicillin-Clavulanate",
             "Amoxicillin-Clavulanic acid",
             "Antibiotic-Penicillin",
             ["resistant infections", "bite wounds", "sinusitis",
              "UTI"],
             ["penicillin allergy",
              "history of cholestatic jaundice"],
             {"adult": "625mg every 8 hours"}),
            ("Nystatin", "Nystatin", "Antifungal",
             ["oral thrush", "intestinal candidiasis"],
             [],
             {"oral thrush": "100,000 units 4 times daily for "
              "7 days"}),
            ("Silver Sulfadiazine", "Silver sulfadiazine",
             "Topical antimicrobial",
             ["burns", "wound infection prevention"],
             ["sulfa allergy", "pregnancy near term", "newborns"],
             {"burns": "Apply 1% cream once or twice daily"}),
            ("Tetracycline eye ointment", "Tetracycline",
             "Topical antibiotic",
             ["conjunctivitis", "trachoma",
              "neonatal eye prophylaxis"],
             [],
             {"conjunctivitis": "Apply 1% ointment 3-4 times daily "
              "for 7 days"}),
            ("Permethrin", "Permethrin", "Insecticide/Scabicide",
             ["scabies", "head lice"],
             [],
             {"scabies": "Apply 5% cream, wash off after "
              "8-14 hours",
              "lice": "Apply 1% lotion, wash off after "
              "10 minutes"}),
            ("Benzyl Benzoate", "Benzyl benzoate", "Scabicide",
             ["scabies"],
             ["broken skin"],
             {"scabies": "Apply 25% lotion, wash off after "
              "24 hours, repeat after 24h"}),
            ("Atropine", "Atropine", "Anticholinergic",
             ["bradycardia", "organophosphate poisoning",
              "eye examination"],
             ["glaucoma", "prostatic hypertrophy"],
             {"bradycardia": "0.5-1mg IV",
              "eye": "1-2 drops 1% solution"}),
        ]

        for drug in essential_drugs:
            cursor.execute(
                "INSERT OR IGNORE INTO drugs "
                "(name, generic_name, drug_class, common_uses, "
                "contraindications, common_doses) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    drug[0], drug[1], drug[2],
                    json.dumps(drug[3]),
                    json.dumps(drug[4]),
                    json.dumps(drug[5]),
                ),
            )

        interactions = [
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
             "Disulfiram-like reaction: severe nausea, vomiting, "
             "flushing",
             "No alcohol during treatment and 48 hours after"),
            ("Ciprofloxacin", "Theophylline", "severe",
             "Increased theophylline levels, risk of toxicity",
             "Monitor theophylline levels, consider dose reduction"),
            ("ACE inhibitors", "Potassium supplements", "severe",
             "Risk of dangerous hyperkalemia",
             "Monitor potassium levels closely"),
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
             "Take ciprofloxacin 2 hours before or 6 hours after "
             "antacids"),
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
            ("Azithromycin", "Amiodarone", "severe",
             "Risk of fatal cardiac arrhythmia (QT prolongation)",
             "Avoid combination - use alternative antibiotic"),
            ("Gentamicin", "Furosemide", "severe",
             "Increased risk of ototoxicity and nephrotoxicity",
             "Monitor renal function and hearing, consider "
             "alternatives"),
            ("Fluconazole", "Warfarin", "severe",
             "Greatly increased bleeding risk due to warfarin "
             "metabolism inhibition",
             "Reduce warfarin dose and monitor INR closely"),
            ("Diazepam", "Alcohol", "severe",
             "Severe CNS and respiratory depression, potentially "
             "fatal",
             "Absolutely avoid alcohol with benzodiazepines"),
            ("Erythromycin", "Carbamazepine", "severe",
             "Increased carbamazepine levels leading to toxicity",
             "Use azithromycin instead or monitor carbamazepine "
             "levels"),
            ("Tramadol", "SSRI antidepressants", "severe",
             "Risk of serotonin syndrome - potentially "
             "life-threatening",
             "Avoid combination, use alternative analgesic"),
            ("Metronidazole", "Warfarin", "moderate",
             "Increased anticoagulant effect and bleeding risk",
             "Monitor INR more frequently"),
            ("Ciprofloxacin", "Iron supplements", "moderate",
             "Reduced ciprofloxacin absorption by up to 50%",
             "Take ciprofloxacin 2 hours before or 6 hours after "
             "iron"),
            ("Azithromycin", "Antacids", "moderate",
             "Reduced azithromycin absorption",
             "Take azithromycin 1 hour before or 2 hours after "
             "antacids"),
            ("Prednisolone", "NSAIDs", "moderate",
             "Increased risk of GI ulceration and bleeding",
             "Use gastroprotection (omeprazole) if combination "
             "necessary"),
            ("Glibenclamide", "Fluconazole", "moderate",
             "Increased hypoglycemia risk",
             "Monitor blood glucose closely, consider dose "
             "reduction"),
            ("Phenobarbital", "Doxycycline", "moderate",
             "Reduced doxycycline levels due to enzyme induction",
             "May need increased doxycycline dose or alternative "
             "antibiotic"),
            ("ACE inhibitors", "NSAIDs", "moderate",
             "Reduced antihypertensive effect and increased renal "
             "risk",
             "Use lowest NSAID dose for shortest time, monitor "
             "renal function"),
            ("Metformin", "ACE inhibitors", "mild",
             "ACE inhibitors may slightly increase hypoglycemia risk",
             "Monitor blood glucose, usually beneficial combination"),
            ("Paracetamol", "Alcohol", "mild",
             "Increased risk of liver damage with chronic alcohol use",
             "Limit alcohol intake"),
            ("Omeprazole", "Iron supplements", "mild",
             "Reduced iron absorption",
             "Consider taking iron with vitamin C"),
            ("Metformin", "Vitamin B12", "mild",
             "Long-term use may reduce B12 absorption",
             "Monitor B12 levels yearly"),
            ("Chlorpheniramine", "Alcohol", "mild",
             "Increased drowsiness",
             "Avoid driving, limit alcohol"),
            ("Doxycycline", "Dairy products", "mild",
             "Slightly reduced doxycycline absorption",
             "Take on empty stomach with full glass of water"),
            ("Omeprazole", "Clopidogrel", "moderate",
             "Reduced antiplatelet effect of clopidogrel",
             "Use pantoprazole instead if PPI needed"),
        ]

        for interaction in interactions:
            cursor.execute(
                "INSERT OR IGNORE INTO interactions "
                "(drug1, drug2, severity, description, recommendation) "
                "VALUES (?, ?, ?, ?, ?)",
                interaction,
            )
            cursor.execute(
                "INSERT OR IGNORE INTO interactions "
                "(drug1, drug2, severity, description, recommendation) "
                "VALUES (?, ?, ?, ?, ?)",
                (interaction[1], interaction[0], interaction[2],
                 interaction[3], interaction[4]),
            )

        conn.commit()
        logger.info(
            "Populated database with %d drugs and %d interactions",
            len(essential_drugs), len(interactions),
        )

    def check_interactions(
        self, medications: List[str],
    ) -> List[DrugInteraction]:
        """Check for interactions between medications.

        Tries the DDInter API first, then falls back to the local
        database.

        Args:
            medications: Medication names to check (needs >= 2).

        Returns:
            Interactions sorted by severity (severe first).
        """
        if len(medications) < 2:
            return []

        client = self._get_ddinter_client()
        if client:
            try:
                from .ddinter_api import DDInterInteraction
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

        logger.info("Using local drug interaction database")
        return self._check_interactions_local(medications)

    def _check_interactions_local(
        self, medications: List[str],
    ) -> List[DrugInteraction]:
        """Check interactions using the local SQLite database.

        Args:
            medications: Medication names to check.

        Returns:
            Interactions sorted by severity.
        """
        result: List[DrugInteraction] = []
        conn = self._get_connection()
        cursor = conn.cursor()

        meds_normalized = [
            self._normalize_drug_name(m) for m in medications
        ]
        checked_pairs: Set[Tuple[str, str]] = set()

        for i, med1 in enumerate(meds_normalized):
            for med2 in meds_normalized[i + 1:]:
                pair = tuple(sorted([med1, med2]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                interaction = self._find_interaction(cursor, med1, med2)
                if interaction:
                    result.append(interaction)

                class_interaction = self._find_class_interaction(
                    cursor, med1, med2,
                )
                if class_interaction and class_interaction not in result:
                    result.append(class_interaction)

        severity_order = {"severe": 0, "moderate": 1, "mild": 2}
        result.sort(key=lambda x: severity_order.get(x.severity, 3))
        return result

    def _normalize_drug_name(self, name: str) -> str:
        """Normalize a drug name for database lookup.

        Strips dosage information, form suffixes, and whitespace.

        Args:
            name: Raw drug name string.

        Returns:
            Cleaned lowercase drug name.
        """
        name = name.lower().strip()
        name = re.sub(r"\d+\s*mg", "", name)
        name = re.sub(r"\d+\s*ml", "", name)
        name = re.sub(r"\d+%", "", name)

        suffixes = [
            "tablet", "tablets", "capsule", "capsules", "cream",
            "ointment", "syrup", "injection", "solution", "suspension",
        ]
        for suffix in suffixes:
            name = name.replace(suffix, "")

        return name.strip()

    def _find_interaction(
        self, cursor: sqlite3.Cursor, drug1: str, drug2: str,
    ) -> Optional[DrugInteraction]:
        """Find a direct interaction between two normalized drug names.

        Args:
            cursor: Active database cursor.
            drug1: First normalized drug name.
            drug2: Second normalized drug name.

        Returns:
            ``DrugInteraction`` if found, else ``None``.
        """
        cursor.execute(
            "SELECT severity, description, recommendation "
            "FROM interactions "
            "WHERE LOWER(drug1) = ? AND LOWER(drug2) = ?",
            (drug1, drug2),
        )
        row = cursor.fetchone()
        if row:
            return DrugInteraction(
                drugs=(drug1, drug2),
                severity=row["severity"],
                description=row["description"],
                recommendation=row["recommendation"],
            )

        cursor.execute(
            "SELECT drug1, drug2, severity, description, recommendation "
            "FROM interactions "
            "WHERE LOWER(drug1) LIKE ? AND LOWER(drug2) LIKE ?",
            (f"%{drug1}%", f"%{drug2}%"),
        )
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
        self, cursor: sqlite3.Cursor, drug1: str, drug2: str,
    ) -> Optional[DrugInteraction]:
        """Find an interaction based on drug pharmacological classes.

        Args:
            cursor: Active database cursor.
            drug1: First normalized drug name.
            drug2: Second normalized drug name.

        Returns:
            ``DrugInteraction`` if a class-level match is found.
        """
        cursor.execute(
            "SELECT name, drug_class FROM drugs "
            "WHERE LOWER(name) LIKE ? OR LOWER(generic_name) LIKE ?",
            (f"%{drug1}%", f"%{drug1}%"),
        )
        row1 = cursor.fetchone()

        cursor.execute(
            "SELECT name, drug_class FROM drugs "
            "WHERE LOWER(name) LIKE ? OR LOWER(generic_name) LIKE ?",
            (f"%{drug2}%", f"%{drug2}%"),
        )
        row2 = cursor.fetchone()

        if not row1 or not row2:
            return None

        class1 = row1["drug_class"]
        class2 = row2["drug_class"]
        if not class1 or not class2:
            return None

        cursor.execute(
            "SELECT severity, description, recommendation "
            "FROM interactions "
            "WHERE LOWER(drug1) LIKE ? AND LOWER(drug2) LIKE ?",
            (f"%{class1.lower()}%", f"%{class2.lower()}%"),
        )
        row = cursor.fetchone()
        if row:
            return DrugInteraction(
                drugs=(drug1, drug2),
                severity=row["severity"],
                description=(
                    f"{row['description']} "
                    f"(class interaction: {class1} + {class2})"
                ),
                recommendation=row["recommendation"],
            )

        return None

    def get_drug_info(self, drug_name: str) -> Optional[DrugInfo]:
        """Look up reference information for a drug.

        Args:
            drug_name: Drug name (partial match supported).

        Returns:
            ``DrugInfo`` if found, else ``None``.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM drugs "
            "WHERE LOWER(name) LIKE ? OR LOWER(generic_name) LIKE ?",
            (f"%{drug_name.lower()}%", f"%{drug_name.lower()}%"),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return DrugInfo(
            name=row["name"],
            generic_name=row["generic_name"],
            drug_class=row["drug_class"],
            common_uses=(
                json.loads(row["common_uses"])
                if row["common_uses"] else []
            ),
            contraindications=(
                json.loads(row["contraindications"])
                if row["contraindications"] else []
            ),
            common_doses=(
                json.loads(row["common_doses"])
                if row["common_doses"] else {}
            ),
        )

    def search_drugs(self, query: str, limit: int = 10) -> List[DrugInfo]:
        """Search for drugs matching a free-text query.

        Args:
            query: Search string (matches name, generic name, or class).
            limit: Maximum number of results.

        Returns:
            List of matching ``DrugInfo`` objects.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM drugs "
            "WHERE LOWER(name) LIKE ? OR LOWER(generic_name) LIKE ? "
            "OR LOWER(drug_class) LIKE ? "
            "LIMIT ?",
            (
                f"%{query.lower()}%",
                f"%{query.lower()}%",
                f"%{query.lower()}%",
                limit,
            ),
        )

        results: List[DrugInfo] = []
        for row in cursor.fetchall():
            results.append(DrugInfo(
                name=row["name"],
                generic_name=row["generic_name"],
                drug_class=row["drug_class"],
                common_uses=(
                    json.loads(row["common_uses"])
                    if row["common_uses"] else []
                ),
                contraindications=(
                    json.loads(row["contraindications"])
                    if row["contraindications"] else []
                ),
                common_doses=(
                    json.loads(row["common_doses"])
                    if row["common_doses"] else {}
                ),
            ))
        return results

    def check_contraindications(
        self, drug_name: str, conditions: List[str],
    ) -> List[str]:
        """Check if a drug is contraindicated for given conditions.

        Args:
            drug_name: Drug to check.
            conditions: Patient conditions to screen against.

        Returns:
            List of matching contraindication strings.
        """
        drug_info = self.get_drug_info(drug_name)
        if not drug_info:
            return []

        matches: List[str] = []
        for condition in conditions:
            condition_lower = condition.lower()
            for contra in drug_info.contraindications:
                if (
                    condition_lower in contra.lower()
                    or contra.lower() in condition_lower
                ):
                    matches.append(contra)
        return matches

    def get_dosage(
        self, drug_name: str, indication: Optional[str] = None,
    ) -> Optional[str]:
        """Look up dosage information for a drug.

        Args:
            drug_name: Drug to look up.
            indication: Optional indication to match against.

        Returns:
            Dosage string, or ``None`` if not found.
        """
        drug_info = self.get_drug_info(drug_name)
        if not drug_info or not drug_info.common_doses:
            return None

        if indication:
            indication_lower = indication.lower()
            for ind, dose in drug_info.common_doses.items():
                if (
                    indication_lower in ind.lower()
                    or ind.lower() in indication_lower
                ):
                    return dose

        return next(iter(drug_info.common_doses.values()), None)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
