"""
Triage agent for diagnosis and treatment reasoning.

Combines symptom information and visual findings to generate
diagnosis and treatment recommendations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class Medication:
    """Medication in a treatment plan."""
    name: str
    dosage: str
    duration: Optional[str] = None
    route: str = "oral"  # oral, topical, IM, IV
    frequency: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Diagnosis:
    """Diagnosis result."""
    condition: str
    confidence: float  # 0-1
    supporting_evidence: List[str] = field(default_factory=list)
    differential_diagnoses: List[str] = field(default_factory=list)
    icd_code: Optional[str] = None


@dataclass
class TreatmentPlan:
    """Treatment plan for a diagnosis."""
    medications: List[Medication] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    follow_up_days: Optional[int] = None
    warning_signs: List[str] = field(default_factory=list)
    requires_referral: bool = False
    referral_reason: Optional[str] = None


class TriageAgent:
    """
    Medical reasoning agent for diagnosis and treatment.

    Supports multiple backends:
    - Ollama (local, recommended)
    - HuggingFace (MedGemma, requires access)
    - Rule-based fallback
    """

    def __init__(self, config, vision_analyzer=None):
        """Initialize the triage agent."""
        self.config = config
        self.vision = vision_analyzer
        self._ollama_client = None
        self._model = None
        self._tokenizer = None
        self._backend = None

    def _init_backend(self):
        """Initialize the appropriate backend."""
        if self._backend is not None:
            return

        # Try Ollama first
        if self.config.backend == "ollama":
            try:
                from .ollama_client import OllamaClient, MEDICAL_SYSTEM_PROMPT
                self._ollama_client = OllamaClient(self.config.ollama_host)

                if self._ollama_client.is_available():
                    if self._ollama_client.has_model(self.config.ollama_model):
                        self._backend = "ollama"
                        logger.info(f"Using Ollama backend with model: {self.config.ollama_model}")
                        return
                    else:
                        logger.warning(f"Model {self.config.ollama_model} not found in Ollama")
                        print(f"Model '{self.config.ollama_model}' not found. Pull it with:")
                        print(f"  ollama pull {self.config.ollama_model}")
            except Exception as e:
                logger.warning(f"Ollama init failed: {e}")

        # Try HuggingFace
        if self.config.backend in ["huggingface", "ollama"]:  # ollama falls back to HF
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                model_id = self.config.medgemma_model_id
                logger.info(f"Trying HuggingFace model: {model_id}")

                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=True
                )

                if self.config.use_4bit_quantization and self.config.device == "cuda":
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=quant_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                else:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                        trust_remote_code=True,
                    )
                    self._model.to(self.config.device)

                self._backend = "huggingface"
                logger.info("Using HuggingFace backend")
                return
            except Exception as e:
                logger.warning(f"HuggingFace init failed: {e}")

        # Fall back to rule-based
        self._backend = "rule_based"
        logger.info("Using rule-based backend")

    def _generate_response(self, prompt: str) -> str:
        """Generate a response from the model."""
        self._init_backend()

        if self._backend == "rule_based":
            return self._rule_based_response(prompt)

        if self._backend == "ollama":
            return self._ollama_generate(prompt)

        if self._backend == "huggingface":
            return self._huggingface_generate(prompt)

        return self._rule_based_response(prompt)

    def _ollama_generate(self, prompt: str) -> str:
        """Generate using Ollama."""
        from .ollama_client import MEDICAL_SYSTEM_PROMPT

        try:
            response = self._ollama_client.generate(
                model=self.config.ollama_model,
                prompt=prompt,
                system=MEDICAL_SYSTEM_PROMPT,
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
            )
            return response
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return self._rule_based_response(prompt)

    def _huggingface_generate(self, prompt: str) -> str:
        """Generate using HuggingFace model."""
        try:
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                response = response[len(prompt):].strip()
            return response
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            return self._rule_based_response(prompt)

    def diagnose_and_treat(
        self,
        symptoms: str,
        visual_findings: List[str],
        patient_age: Optional[str] = None,
        patient_conditions: Optional[List[str]] = None,
    ) -> Tuple[Diagnosis, TreatmentPlan]:
        """
        Generate diagnosis and treatment plan.

        Args:
            symptoms: Text description of symptoms
            visual_findings: List of findings from image analysis
            patient_age: Optional age information (e.g., "adult", "child 5 years")
            patient_conditions: Optional list of known conditions

        Returns:
            Tuple of (Diagnosis, TreatmentPlan)
        """
        # Build comprehensive prompt
        prompt = self._build_diagnosis_prompt(
            symptoms, visual_findings, patient_age, patient_conditions
        )

        # Generate response
        response = self._generate_response(prompt)

        # Parse response into structured output
        diagnosis = self._parse_diagnosis(response, symptoms, visual_findings)
        treatment = self._parse_treatment(response, diagnosis)

        return diagnosis, treatment

    def _build_diagnosis_prompt(
        self,
        symptoms: str,
        visual_findings: List[str],
        patient_age: Optional[str],
        patient_conditions: Optional[List[str]],
    ) -> str:
        """Build the diagnosis prompt requesting structured output."""
        prompt = """You are a medical decision support system helping a Community Health Worker.
Based on the patient information, provide diagnosis and treatment in the EXACT format shown below.

PATIENT INFORMATION:
"""
        if patient_age:
            prompt += f"Age: {patient_age}\n"

        if patient_conditions:
            prompt += f"Known conditions: {', '.join(patient_conditions)}\n"

        prompt += f"\nSYMPTOMS:\n{symptoms}\n"

        if visual_findings:
            prompt += f"\nVISUAL FINDINGS:\n"
            for finding in visual_findings:
                prompt += f"- {finding}\n"

        prompt += """
Respond in EXACTLY this format:

DIAGNOSIS: [condition name]
CONFIDENCE: [high/medium/low]
DIFFERENTIALS: [other possible conditions, comma separated]

MEDICATIONS:
- [Drug name]: [specific dosage and frequency]
- [Drug name]: [specific dosage and frequency]

INSTRUCTIONS:
- [instruction 1]
- [instruction 2]

WARNING SIGNS:
- [sign 1]
- [sign 2]

FOLLOW UP: [number] days
REFERRAL: [yes/no] - [reason if yes]"""
        return prompt

    def _parse_diagnosis(
        self,
        response: str,
        symptoms: str,
        visual_findings: List[str],
    ) -> Diagnosis:
        """Parse diagnosis from structured response."""

        # Extract DIAGNOSIS line
        condition = "Unknown condition"
        match = re.search(r'DIAGNOSIS:\s*(.+)', response, re.IGNORECASE)
        if match:
            condition = match.group(1).strip()

        # Extract CONFIDENCE line
        confidence = 0.7
        match = re.search(r'CONFIDENCE:\s*(high|medium|low)', response, re.IGNORECASE)
        if match:
            conf_map = {"high": 0.85, "medium": 0.7, "low": 0.5}
            confidence = conf_map.get(match.group(1).lower(), 0.7)

        # Extract DIFFERENTIALS line
        differentials = []
        match = re.search(r'DIFFERENTIALS:\s*(.+)', response, re.IGNORECASE)
        if match:
            differentials = [d.strip() for d in match.group(1).split(',') if d.strip()]

        # Build evidence list
        evidence = []
        if symptoms:
            evidence.append(f"Symptoms: {symptoms[:100]}")
        evidence.extend(visual_findings[:3])

        return Diagnosis(
            condition=condition,
            confidence=confidence,
            supporting_evidence=evidence,
            differential_diagnoses=differentials,
        )

    def _parse_treatment(self, response: str, diagnosis: Diagnosis) -> TreatmentPlan:
        """Parse treatment plan from structured response."""

        medications = self._parse_medications_section(response)
        instructions = self._parse_list_section(response, "INSTRUCTIONS")
        warning_signs = self._parse_list_section(response, "WARNING SIGNS")

        # Extract follow-up days
        follow_up = 3
        match = re.search(r'FOLLOW\s*UP:\s*(\d+)', response, re.IGNORECASE)
        if match:
            follow_up = int(match.group(1))

        # Extract referral
        needs_referral = False
        referral_reason = None
        match = re.search(r'REFERRAL:\s*(yes|no)\s*[-:]?\s*(.*)', response, re.IGNORECASE)
        if match:
            needs_referral = match.group(1).lower() == "yes"
            if needs_referral and match.group(2):
                referral_reason = match.group(2).strip()

        return TreatmentPlan(
            medications=medications,
            instructions=instructions,
            follow_up_days=follow_up,
            warning_signs=warning_signs,
            requires_referral=needs_referral,
            referral_reason=referral_reason,
        )

    def _parse_medications_section(self, response: str) -> List[Medication]:
        """Parse MEDICATIONS section from structured response."""
        medications = []

        # Find MEDICATIONS section
        match = re.search(
            r'MEDICATIONS:\s*(.*?)(?=\n(?:INSTRUCTIONS|WARNING|FOLLOW|REFERRAL|$))',
            response,
            re.IGNORECASE | re.DOTALL
        )

        if match:
            section = match.group(1)
            lines = section.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Remove bullet points
                line = re.sub(r'^[\-\*\•]\s*', '', line)

                # Parse "Drug: dosage" or "Drug - dosage"
                if ':' in line or ' - ' in line:
                    parts = re.split(r'[:\-]', line, 1)
                    if len(parts) == 2:
                        name = parts[0].strip()
                        dosage = parts[1].strip()
                        if name and dosage:
                            medications.append(Medication(name=name, dosage=dosage))
                elif line:
                    # Just drug name without dosage
                    medications.append(Medication(name=line, dosage="as directed"))

        return medications

    def _parse_list_section(self, response: str, section_name: str) -> List[str]:
        """Parse a list section (INSTRUCTIONS, WARNING SIGNS) from response."""
        items = []

        # Find section
        pattern = rf'{section_name}:\s*(.*?)(?=\n(?:MEDICATIONS|INSTRUCTIONS|WARNING|FOLLOW|REFERRAL|$))'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)

        if match:
            section = match.group(1)
            lines = section.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Remove bullet points
                line = re.sub(r'^[\-\*\•\d\.]+\s*', '', line).strip()
                if line:
                    items.append(line)

        return items

    # Legacy methods for fallback (kept for rule-based mode)
    def _extract_condition(self, response: str) -> str:
        """Extract primary diagnosis from response (legacy fallback)."""
        response_lower = response.lower()

        patterns = [
            r"diagnosis[:\s]+([^\n]+)",
            r"likely[:\s]+([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                condition = match.group(1).strip()
                # Clean up
                condition = condition.split('.')[0]
                condition = condition.split(',')[0]
                return condition.title()

        # Fallback: look for common conditions mentioned
        common_conditions = [
            "malaria", "pneumonia", "diarrhea", "skin infection",
            "respiratory infection", "urinary tract infection",
            "measles", "ringworm", "wound infection", "dehydration",
            "fever", "gastroenteritis", "conjunctivitis", "otitis media",
        ]

        for condition in common_conditions:
            if condition in response_lower:
                return condition.title()

        return "Unspecified illness"

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence level from response."""
        response_lower = response.lower()

        if "high confidence" in response_lower or "confident" in response_lower:
            return 0.85
        elif "medium confidence" in response_lower or "moderate" in response_lower:
            return 0.65
        elif "low confidence" in response_lower or "uncertain" in response_lower:
            return 0.45

        # Look for percentage
        match = re.search(r"(\d+)%", response)
        if match:
            return int(match.group(1)) / 100

        return 0.7  # Default medium confidence

    def _extract_differentials(self, response: str) -> List[str]:
        """Extract differential diagnoses from response."""
        differentials = []

        # Look for differential section
        diff_section = re.search(
            r"differential[s]?[:\s]*(.*?)(?=\n\n|treatment|referral|$)",
            response.lower(),
            re.DOTALL
        )

        if diff_section:
            text = diff_section.group(1)
            # Split by common delimiters
            items = re.split(r'[\n\-\*\d\.]+', text)
            for item in items:
                item = item.strip()
                if item and len(item) > 3:
                    differentials.append(item.title())

        return differentials[:5]  # Limit to 5

    def _extract_medications(self, response: str) -> List[Medication]:
        """Extract medications from response."""
        medications = []
        found_meds = set()  # Avoid duplicates

        # Known medications with their search patterns
        known_meds = {
            "paracetamol": "Paracetamol",
            "acetaminophen": "Paracetamol",
            "amoxicillin": "Amoxicillin",
            "metronidazole": "Metronidazole",
            "oral rehydration": "ORS",
            "ors": "ORS",
            "zinc": "Zinc",
            "vitamin a": "Vitamin A",
            "artemether": "Artemether-Lumefantrine",
            "lumefantrine": "Artemether-Lumefantrine",
            "coartem": "Artemether-Lumefantrine",
            "ibuprofen": "Ibuprofen",
            "clotrimazole": "Clotrimazole cream",
            "hydrocortisone": "Hydrocortisone cream",
            "cotrimoxazole": "Cotrimoxazole",
            "doxycycline": "Doxycycline",
            "azithromycin": "Azithromycin",
            "ciprofloxacin": "Ciprofloxacin",
            "prednisolone": "Prednisolone",
            "salbutamol": "Salbutamol inhaler",
            "chlorpheniramine": "Chlorpheniramine",
            "antifungal": "Topical antifungal",
            "antibiotic ointment": "Antibiotic ointment",
        }

        response_lower = response.lower()

        # First, try to find medication lines with dosages
        # Look for patterns like "Drug: dosage" or "Drug - dosage" or "• Drug dosage"
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower().strip()

            for med_key, med_name in known_meds.items():
                if med_key in line_lower and med_name not in found_meds:
                    # Try to extract dosage from the same line
                    dosage = self._extract_dosage_from_line(line, med_key)
                    medications.append(Medication(
                        name=med_name,
                        dosage=dosage,
                    ))
                    found_meds.add(med_name)
                    break

        # If no medications found, provide supportive care
        if not medications and "supportive" in response_lower:
            medications.append(Medication(
                name="Supportive care",
                dosage="Rest, fluids, monitor symptoms",
            ))

        return medications

    def _extract_dosage_from_line(self, line: str, med_keyword: str) -> str:
        """Extract dosage information from a line containing a medication."""
        line_lower = line.lower()

        # Find where the medication name ends
        med_pos = line_lower.find(med_keyword)
        if med_pos == -1:
            return "as directed"

        # Get text after the medication name
        after_med = line[med_pos + len(med_keyword):].strip()

        # Remove common prefixes
        after_med = re.sub(r'^[\s:\-\*\•]+', '', after_med).strip()

        # Look for dosage patterns
        dosage_patterns = [
            # "500mg every 6 hours" or "500 mg three times daily"
            r'(\d+\s*(?:mg|ml|g|mcg|iu|units?)[^.]*(?:daily|hourly|hours|times|twice|once|per day|every)[^.]*)',
            # "4 tablets twice daily"
            r'(\d+\s*tablets?[^.]*(?:daily|times|twice|once)[^.]*)',
            # "Apply twice daily" or "Take as needed"
            r'((?:apply|take|give|use)\s+[^.]{5,40})',
            # "20mg daily for 10 days"
            r'(\d+\s*(?:mg|ml)[^.]*for\s+\d+\s*days[^.]*)',
            # Just grab numbers with units
            r'(\d+\s*(?:mg|ml|g|tablets?|capsules?|puffs?)[^,.\n]{0,30})',
        ]

        for pattern in dosage_patterns:
            match = re.search(pattern, after_med, re.IGNORECASE)
            if match:
                dosage = match.group(1).strip()
                # Clean up
                dosage = re.sub(r'\s+', ' ', dosage)
                if len(dosage) > 5:
                    return dosage

        # Check the whole line for dosage info
        for pattern in dosage_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                dosage = match.group(1).strip()
                dosage = re.sub(r'\s+', ' ', dosage)
                if len(dosage) > 5:
                    return dosage

        # Default dosages for common medications
        defaults = {
            "paracetamol": "500-1000mg every 4-6 hours (max 4g/day)",
            "acetaminophen": "500-1000mg every 4-6 hours (max 4g/day)",
            "amoxicillin": "500mg every 8 hours for 5-7 days",
            "ibuprofen": "400mg every 6-8 hours with food",
            "oral rehydration": "As needed for hydration",
            "ors": "As needed for hydration",
            "zinc": "20mg daily for 10-14 days",
            "vitamin a": "200,000 IU single dose",
            "artemether": "4 tablets at 0, 8, 24, 36, 48, 60 hours",
            "metronidazole": "400mg every 8 hours for 5-7 days",
            "clotrimazole": "Apply twice daily for 2-4 weeks",
            "hydrocortisone": "Apply thin layer 1-2 times daily",
        }

        return defaults.get(med_keyword, "as directed")

    def _extract_instructions(self, response: str) -> List[str]:
        """Extract non-medication instructions from response."""
        instructions = []

        # Look for instruction patterns
        instruction_keywords = [
            "rest", "fluid", "drink", "hydrat", "clean", "wash",
            "monitor", "observe", "return", "avoid", "keep",
        ]

        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if any(kw in line_lower for kw in instruction_keywords):
                # Clean up the line
                line = re.sub(r'^[\-\*\d\.]+\s*', '', line).strip()
                if line and len(line) > 10:
                    instructions.append(line)

        # Add default instructions if none found
        if not instructions:
            instructions = [
                "Ensure adequate rest",
                "Maintain hydration with clean water",
                "Return if symptoms worsen",
            ]

        return instructions[:5]

    def _extract_warning_signs(self, response: str) -> List[str]:
        """Extract warning signs to watch for."""
        warning_signs = []

        # Look for warning section
        warning_section = re.search(
            r"warning[s]?[:\s]*(.*?)(?=\n\n|follow|referral|$)",
            response.lower(),
            re.DOTALL
        )

        if warning_section:
            text = warning_section.group(1)
            items = re.split(r'[\n\-\*]+', text)
            for item in items:
                item = item.strip()
                if item and len(item) > 5:
                    warning_signs.append(item)

        # Add common warning signs
        default_warnings = [
            "High fever that doesn't respond to paracetamol",
            "Difficulty breathing",
            "Unable to drink or eat",
            "Altered consciousness or confusion",
            "Severe vomiting or diarrhea",
        ]

        if not warning_signs:
            warning_signs = default_warnings[:3]

        return warning_signs[:5]

    def _extract_referral(self, response: str) -> Tuple[bool, Optional[str]]:
        """Determine if referral is needed."""
        response_lower = response.lower()

        # Look for explicit referral indicators
        referral_yes = [
            "refer", "hospital", "emergency", "urgent",
            "cannot be managed", "beyond scope",
        ]

        referral_no = [
            "no referral", "can be managed", "does not require",
            "outpatient", "health post level",
        ]

        # Check for no referral first
        for phrase in referral_no:
            if phrase in response_lower:
                return False, None

        # Check for referral needed
        for phrase in referral_yes:
            if phrase in response_lower:
                # Try to extract reason
                match = re.search(
                    rf"{phrase}[:\s]*([^\n\.]+)",
                    response_lower
                )
                reason = match.group(1).strip() if match else "Requires higher level care"
                return True, reason

        return False, None

    def _determine_follow_up(self, condition: str) -> int:
        """Determine appropriate follow-up days based on condition."""
        condition_lower = condition.lower()

        # Shorter follow-up for acute conditions
        short_follow_up = ["fever", "infection", "malaria", "diarrhea"]
        for cond in short_follow_up:
            if cond in condition_lower:
                return 2

        # Medium follow-up
        medium_follow_up = ["skin", "wound", "rash"]
        for cond in medium_follow_up:
            if cond in condition_lower:
                return 5

        # Default follow-up
        return 3

    def _rule_based_response(self, prompt: str) -> str:
        """Generate rule-based response when model unavailable."""
        prompt_lower = prompt.lower()

        # Determine likely condition based on keywords
        if "fever" in prompt_lower and "rash" in prompt_lower:
            return self._measles_response()
        elif "fever" in prompt_lower and "headache" in prompt_lower:
            return self._malaria_response()
        elif "diarrhea" in prompt_lower or "vomit" in prompt_lower:
            return self._gastroenteritis_response()
        elif "cough" in prompt_lower and "fever" in prompt_lower:
            return self._respiratory_response()
        elif "wound" in prompt_lower or "cut" in prompt_lower:
            return self._wound_response()
        elif "skin" in prompt_lower or "rash" in prompt_lower:
            return self._skin_response()
        else:
            return self._general_response()

    def _measles_response(self) -> str:
        return """
1. DIAGNOSIS
- Primary diagnosis: Suspected Measles (medium confidence)
- Supporting evidence: Fever with rash
- Differential diagnoses: Other viral exanthems, rubella, scarlet fever

2. TREATMENT PLAN
- Medications:
  - Vitamin A: 200,000 IU single dose (100,000 IU if under 1 year)
  - Paracetamol: 500mg every 6 hours for fever (weight-based for children)
- Instructions:
  - Isolate patient to prevent spread
  - Ensure adequate hydration
  - Keep in dim lighting if eyes are affected
- Warning signs: Difficulty breathing, ear discharge, altered consciousness

3. REFERRAL
- Hospital referral: Consider if severe symptoms or complications
"""

    def _malaria_response(self) -> str:
        return """
1. DIAGNOSIS
- Primary diagnosis: Suspected Malaria (medium confidence)
- Supporting evidence: Fever, headache
- Differential diagnoses: Typhoid, viral illness, urinary tract infection

2. TREATMENT PLAN
- Medications:
  - Artemether-Lumefantrine: 4 tablets at 0, 8, 24, 36, 48, 60 hours (adult dose)
  - Paracetamol: 500-1000mg every 6 hours for fever
- Instructions:
  - Complete full course of antimalarial
  - Maintain hydration
  - Use bed net to prevent transmission
- Warning signs: Severe vomiting, altered consciousness, jaundice, dark urine

3. REFERRAL
- Hospital referral: Yes if rapid diagnostic test is positive and symptoms are severe
"""

    def _gastroenteritis_response(self) -> str:
        return """
1. DIAGNOSIS
- Primary diagnosis: Acute Gastroenteritis (high confidence)
- Supporting evidence: Diarrhea and/or vomiting
- Differential diagnoses: Food poisoning, cholera, dysentery

2. TREATMENT PLAN
- Medications:
  - Oral Rehydration Salts: As needed for hydration
  - Zinc: 20mg daily for 10-14 days (children)
- Instructions:
  - Continue feeding/breastfeeding
  - Small frequent sips of ORS
  - Monitor for dehydration signs
- Warning signs: Bloody stool, unable to drink, sunken eyes, no urine for 6 hours

3. REFERRAL
- Hospital referral: Yes if severe dehydration or bloody diarrhea
"""

    def _respiratory_response(self) -> str:
        return """
1. DIAGNOSIS
- Primary diagnosis: Acute Respiratory Infection (medium confidence)
- Supporting evidence: Cough and fever
- Differential diagnoses: Pneumonia, bronchitis, tuberculosis

2. TREATMENT PLAN
- Medications:
  - Paracetamol: 500mg every 6 hours for fever
  - Amoxicillin: 500mg every 8 hours for 5 days (if bacterial suspected)
- Instructions:
  - Rest and adequate fluids
  - Steam inhalation for congestion
  - Monitor breathing rate
- Warning signs: Fast breathing, chest indrawing, unable to drink, cyanosis

3. REFERRAL
- Hospital referral: Yes if signs of pneumonia or breathing difficulty
"""

    def _wound_response(self) -> str:
        return """
1. DIAGNOSIS
- Primary diagnosis: Wound requiring treatment (high confidence)
- Supporting evidence: Visible wound
- Differential diagnoses: Infected wound, abscess

2. TREATMENT PLAN
- Medications:
  - Wound cleaning with saline or clean water
  - Antibiotic ointment if superficial
  - Amoxicillin: 500mg every 8 hours for 5 days (if infected)
  - Paracetamol: 500mg every 6 hours for pain
- Instructions:
  - Clean wound daily
  - Keep wound dry and covered
  - Check tetanus vaccination status
- Warning signs: Increasing redness, swelling, pus, red streaks, fever

3. REFERRAL
- Hospital referral: Yes if deep wound, needs suturing, or signs of severe infection
"""

    def _skin_response(self) -> str:
        return """
1. DIAGNOSIS
- Primary diagnosis: Skin condition requiring treatment (medium confidence)
- Supporting evidence: Visible skin changes
- Differential diagnoses: Fungal infection, eczema, allergic reaction, scabies

2. TREATMENT PLAN
- Medications:
  - Clotrimazole cream: Apply twice daily for 2-4 weeks (if fungal suspected)
  - Hydrocortisone cream: Apply twice daily for itch (if allergic/eczema)
  - Chlorpheniramine: 4mg every 8 hours for itch
- Instructions:
  - Keep area clean and dry
  - Avoid scratching
  - Monitor for spread or worsening
- Warning signs: Rapid spread, fever, blistering, facial involvement

3. REFERRAL
- Hospital referral: Consider if not improving after 1-2 weeks of treatment
"""

    def _general_response(self) -> str:
        return """
1. DIAGNOSIS
- Primary diagnosis: Unspecified illness requiring evaluation (low confidence)
- Supporting evidence: Symptoms reported
- Differential diagnoses: Multiple conditions possible

2. TREATMENT PLAN
- Medications:
  - Paracetamol: 500mg every 6 hours for fever/pain as needed
  - Supportive care
- Instructions:
  - Rest and adequate hydration
  - Monitor symptoms
  - Return if no improvement in 48 hours
- Warning signs: High fever, difficulty breathing, altered consciousness

3. REFERRAL
- Hospital referral: Consider if symptoms worsen or diagnosis unclear
"""
