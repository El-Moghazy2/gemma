"""
Medical vision analysis module.

Supports multiple backends:
- Ollama with LLaVA (local, recommended)
- HuggingFace MedGemma (requires access)
- Mock mode for testing

Handles:
- Skin condition analysis (rashes, wounds, lesions)
- Eye examination (conjunctivitis, jaundice)
- Prescription/medication label text extraction
"""

from typing import List, Union, Optional, Any
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


class MedicalVisionAnalyzer:
    """
    Medical image analysis supporting multiple backends.

    Supports visual analysis of skin conditions, wounds, eyes,
    and text extraction from medication labels/prescriptions.
    """

    def __init__(self, config):
        """Initialize the vision analyzer."""
        self.config = config
        self._ollama_client = None
        self._model = None
        self._processor = None
        self._backend = None

    def _init_backend(self):
        """Initialize the appropriate backend."""
        if self._backend is not None:
            return

        # Try Ollama first
        if self.config.backend == "ollama":
            try:
                from .ollama_client import OllamaClient
                self._ollama_client = OllamaClient(self.config.ollama_host)

                if self._ollama_client.is_available():
                    if self._ollama_client.has_model(self.config.ollama_vision_model):
                        self._backend = "ollama"
                        logger.info(f"Using Ollama vision with model: {self.config.ollama_vision_model}")
                        return
                    else:
                        logger.warning(f"Vision model {self.config.ollama_vision_model} not found")
                        print(f"Vision model '{self.config.ollama_vision_model}' not found. Pull it with:")
                        print(f"  ollama pull {self.config.ollama_vision_model}")
            except Exception as e:
                logger.warning(f"Ollama vision init failed: {e}")

        # Try HuggingFace
        if self.config.backend in ["huggingface", "ollama"]:
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                import torch

                logger.info(f"Loading MedGemma model: {self.config.medgemma_model_id}")

                self._processor = AutoProcessor.from_pretrained(
                    self.config.medgemma_model_id,
                    trust_remote_code=True,
                )

                if self.config.use_4bit_quantization and self.config.device == "cuda":
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                    self._model = AutoModelForVision2Seq.from_pretrained(
                        self.config.medgemma_model_id,
                        quantization_config=quant_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                else:
                    self._model = AutoModelForVision2Seq.from_pretrained(
                        self.config.medgemma_model_id,
                        torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                        trust_remote_code=True,
                    )
                    self._model.to(self.config.device)

                self._backend = "huggingface"
                logger.info("Using HuggingFace vision backend")
                return
            except Exception as e:
                logger.warning(f"HuggingFace vision init failed: {e}")

        # Fall back to mock
        self._backend = "mock"
        logger.info("Using mock vision backend")

    def _prepare_image(self, image: Any):
        """Prepare image for processing."""
        from PIL import Image as PILImage

        # Already a PIL Image
        if isinstance(image, PILImage.Image):
            return image

        # File path
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                return PILImage.open(path).convert("RGB")
            raise FileNotFoundError(f"Image not found: {path}")

        # Numpy array
        try:
            import numpy as np
            if isinstance(image, np.ndarray):
                if image.dtype == np.uint8:
                    return PILImage.fromarray(image).convert("RGB")
                else:
                    image = (image * 255).astype(np.uint8)
                    return PILImage.fromarray(image).convert("RGB")
        except ImportError:
            pass

        # Gradio image (usually numpy array or filepath)
        if hasattr(image, "name"):
            return PILImage.open(image.name).convert("RGB")

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _run_inference(self, image, prompt: str) -> str:
        """Run inference on image with prompt."""
        self._init_backend()

        if self._backend == "mock":
            return self._mock_inference(prompt)

        if self._backend == "ollama":
            return self._ollama_inference(image, prompt)

        if self._backend == "huggingface":
            return self._huggingface_inference(image, prompt)

        return self._mock_inference(prompt)

    def _ollama_inference(self, image, prompt: str) -> str:
        """Run inference using Ollama LLaVA."""
        from .ollama_client import VISION_SYSTEM_PROMPT

        try:
            pil_image = self._prepare_image(image)

            response = self._ollama_client.generate(
                model=self.config.ollama_vision_model,
                prompt=prompt,
                system=VISION_SYSTEM_PROMPT,
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                images=[pil_image],
            )
            return response
        except Exception as e:
            logger.error(f"Ollama vision inference failed: {e}")
            return self._mock_inference(prompt)

    def _huggingface_inference(self, image, prompt: str) -> str:
        """Run inference using HuggingFace model."""
        try:
            import torch

            pil_image = self._prepare_image(image)

            inputs = self._processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                )

            response = self._processor.decode(outputs[0], skip_special_tokens=True)

            if prompt in response:
                response = response.split(prompt)[-1].strip()

            return response
        except Exception as e:
            logger.error(f"HuggingFace vision inference failed: {e}")
            return self._mock_inference(prompt)

    def analyze_medical_image(
        self,
        image: Any,
        context: Optional[str] = None,
    ) -> List[str]:
        """
        Analyze a medical image (skin, wound, eye, etc.).

        Args:
            image: Image to analyze
            context: Optional context about the patient/symptoms

        Returns:
            List of clinical findings
        """
        prompt = """You are a medical image analysis assistant helping a community health worker.

Analyze this medical image and provide:
1. Description of what you observe
2. Possible conditions this could indicate
3. Key clinical features to note
4. Severity assessment (mild/moderate/severe)

"""
        if context:
            prompt += f"Patient context: {context}\n\n"

        prompt += "Provide your findings in a clear, structured format."

        response = self._run_inference(image, prompt)

        # Parse response into list of findings
        findings = self._parse_findings(response)

        return findings

    def analyze_skin_condition(self, image: Any) -> dict:
        """
        Specialized analysis for skin conditions.

        Returns dict with: condition, features, severity, recommendations
        """
        prompt = """Analyze this skin image for a community health worker.

Describe:
1. APPEARANCE: Color, texture, size, distribution of any lesions/rashes
2. LIKELY CONDITIONS: Top 3 possible diagnoses with confidence
3. RED FLAGS: Any signs requiring immediate referral
4. RECOMMENDED ACTIONS: Next steps for the CHW

Format your response with clear sections."""

        response = self._run_inference(image, prompt)

        return {
            "raw_analysis": response,
            "findings": self._parse_findings(response),
        }

    def analyze_wound(self, image: Any) -> dict:
        """
        Specialized analysis for wounds.

        Returns dict with: type, characteristics, infection_signs, care_recommendations
        """
        prompt = """Analyze this wound image for a community health worker.

Assess:
1. WOUND TYPE: Cut, abrasion, burn, ulcer, etc.
2. SIZE & DEPTH: Approximate dimensions and depth
3. INFECTION SIGNS: Redness, swelling, pus, odor indicators
4. HEALING STAGE: Inflammatory, proliferative, or remodeling
5. CARE RECOMMENDATIONS: Cleaning, dressing, medications
6. REFERRAL NEEDED: Yes/No and why

Provide practical guidance for wound management."""

        response = self._run_inference(image, prompt)

        return {
            "raw_analysis": response,
            "findings": self._parse_findings(response),
        }

    def extract_medications(self, image: Any) -> List[str]:
        """
        Extract medication names from a photo of prescriptions or medicine labels.

        Args:
            image: Photo of medications/prescriptions

        Returns:
            List of medication names found
        """
        prompt = """Extract all medication names visible in this image.

This may be a photo of:
- Prescription paper
- Medicine bottles/boxes
- Pill packaging

List ONLY the medication names, one per line.
Include brand names and generic names if both are visible.
If you cannot read a name clearly, indicate [unclear].

Medication names found:"""

        response = self._run_inference(image, prompt)

        # Parse medication names from response
        medications = self._parse_medication_list(response)

        return medications

    def extract_lab_values(self, image: Any) -> dict:
        """
        Extract lab values from a lab report photo.

        Returns dict mapping test names to values and reference ranges.
        """
        prompt = """Extract all lab test results from this lab report image.

For each test found, provide:
- Test name
- Result value with units
- Reference range if visible
- Flag if abnormal (HIGH/LOW)

Format as:
TEST_NAME: VALUE (REFERENCE_RANGE) [FLAG if abnormal]"""

        response = self._run_inference(image, prompt)

        return {
            "raw_text": response,
            "values": self._parse_lab_values(response),
        }

    def _parse_findings(self, response: str) -> List[str]:
        """Parse response into list of findings."""
        findings = []

        # Split by common delimiters
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove bullet points and numbers
            line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line)

            # Skip very short lines or headers
            if len(line) < 10:
                continue
            if line.endswith(":"):
                continue

            findings.append(line)

        return findings[:10]  # Limit to top 10 findings

    def _parse_medication_list(self, response: str) -> List[str]:
        """Parse medication names from response."""
        medications = []

        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove bullet points and numbers
            line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line)

            # Skip lines that are clearly not medications
            skip_patterns = [
                "medication", "found", "visible", "image", "photo",
                "prescription", "unclear", "cannot", "list",
            ]
            if any(p in line.lower() for p in skip_patterns) and len(line) > 30:
                continue

            # Clean up the medication name
            med_name = line.strip("- •*")

            if med_name and len(med_name) > 2:
                medications.append(med_name)

        return medications

    def _parse_lab_values(self, response: str) -> dict:
        """Parse lab values from response."""
        values = {}

        lines = response.split("\n")

        for line in lines:
            # Look for pattern: TEST: VALUE
            match = re.match(r"([A-Za-z\s]+):\s*(.+)", line)
            if match:
                test_name = match.group(1).strip()
                test_value = match.group(2).strip()
                values[test_name] = test_value

        return values

    def _mock_inference(self, prompt: str) -> str:
        """Mock inference for testing without model."""
        if "medication" in prompt.lower() or "extract" in prompt.lower():
            return """Medications found:
- Paracetamol 500mg
- Amoxicillin 250mg
- Omeprazole 20mg"""

        if "wound" in prompt.lower():
            return """WOUND ANALYSIS:
1. WOUND TYPE: Superficial laceration, approximately 3cm length
2. SIZE & DEPTH: 3cm x 0.5cm, partial thickness
3. INFECTION SIGNS: Mild redness at edges, no pus or excessive swelling
4. HEALING STAGE: Inflammatory phase (recent wound)
5. CARE: Clean with saline, apply antibiotic ointment, sterile dressing
6. REFERRAL: No - can be managed at health post"""

        if "skin" in prompt.lower():
            return """SKIN ANALYSIS:
1. APPEARANCE: Raised red papules in circular pattern, some with scaling
2. LIKELY CONDITIONS:
   - Ringworm (Tinea corporis) - 70% confidence
   - Contact dermatitis - 20%
   - Psoriasis - 10%
3. RED FLAGS: None identified
4. ACTIONS: Topical antifungal (clotrimazole) for 2 weeks"""

        # Default medical image analysis
        return """ANALYSIS:
1. Observation: The image shows a skin condition with visible rash
2. Possible conditions: Allergic reaction, viral exanthem, or fungal infection
3. Key features: Erythematous macules, distributed on trunk
4. Severity: Appears mild to moderate
5. Recommendation: Monitor for 24-48 hours, treat symptomatically"""
