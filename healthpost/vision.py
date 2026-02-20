"""Medical vision analysis module.

Uses MedGemma 1.5 via HuggingFace for all image analysis, including
skin condition analysis, wound assessment, eye examination, and
medication label text extraction.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config

logger = logging.getLogger(__name__)


class MedicalVisionAnalyzer:
    """MedGemma 1.5 medical image analyzer.

    Delegates inference to an :class:`~healthpost.inference_backend.InferenceBackend`.

    Attributes:
        config: Application configuration.
        backend: Inference backend (Unsloth or HuggingFace).
    """

    def __init__(self, config: Config, backend=None) -> None:
        """Initialize the vision analyzer.

        Args:
            config: Application configuration instance.
            backend: Inference backend. If ``None``, one will be created
                lazily via :func:`~healthpost.inference_backend.create_backend`.
        """
        self.config = config
        self._backend = backend

    @property
    def backend(self):
        """Lazily resolved inference backend."""
        if self._backend is None:
            from .inference_backend import create_backend
            self._backend = create_backend(self.config)
        return self._backend

    def _prepare_image(self, image: Any):
        """Convert an image to a PIL ``Image``.

        Args:
            image: PIL Image, file path, numpy array, or Gradio file
                object.

        Returns:
            PIL Image in RGB mode.

        Raises:
            FileNotFoundError: If a path is given but does not exist.
            ValueError: If the image type is unsupported.
        """
        from PIL import Image as PILImage

        source_type = type(image).__name__

        if isinstance(image, PILImage.Image):
            logger.debug("Image source: PIL Image, size=%s", image.size)
            return image

        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                pil_img = PILImage.open(path).convert("RGB")
                logger.debug(
                    "Image source: path (%s), size=%s", path, pil_img.size,
                )
                return pil_img
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            import numpy as np
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_img = PILImage.fromarray(image).convert("RGB")
                logger.debug(
                    "Image source: ndarray, size=%s", pil_img.size,
                )
                return pil_img
        except ImportError:
            pass

        if hasattr(image, "name"):
            pil_img = PILImage.open(image.name).convert("RGB")
            logger.debug(
                "Image source: file object (%s), size=%s",
                image.name, pil_img.size,
            )
            return pil_img

        raise ValueError(f"Unsupported image type: {source_type}")

    def _run_inference(self, image: Any, prompt: str) -> str:
        """Run inference via the configured backend.

        Args:
            image: Image to analyze.
            prompt: Text prompt describing the analysis task.

        Returns:
            Model response text.

        Raises:
            NotImplementedError: If the backend does not support vision.
        """
        if not self.backend.supports_vision:
            logger.info(
                "Backend %s does not support vision — skipping image analysis",
                type(self.backend).__name__,
            )
            raise NotImplementedError(
                f"The current model does not support image analysis. "
                f"Image-based features require a vision-capable model."
            )

        logger.debug(
            "Prompt length=%d, first 100 chars: %.100s", len(prompt), prompt,
        )
        pil_image = self._prepare_image(image)
        return self.backend.generate_with_image(
            pil_image,
            prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
        )

    def analyze_medical_image(
        self,
        image: Any,
        context: Optional[str] = None,
    ) -> List[str]:
        """Analyze a general medical image.

        Args:
            image: Image to analyze (skin, wound, eye, etc.).
            context: Optional patient/symptom context.

        Returns:
            List of clinical finding strings.
        """
        logger.info("Analyzing medical image")

        prompt = (
            "You are a medical image analysis assistant helping a "
            "community health worker.\n\n"
            "Analyze this medical image and provide:\n"
            "1. Description of what you observe\n"
            "2. Possible conditions this could indicate\n"
            "3. Key clinical features to note\n"
            "4. Severity assessment (mild/moderate/severe)\n\n"
        )
        if context:
            prompt += f"Patient context: {context}\n\n"
        prompt += "Provide your findings in a clear, structured format."

        response = self._run_inference(image, prompt)
        findings = self._parse_findings(response)
        logger.info("Medical image analysis complete: %d findings", len(findings))
        return findings

    def analyze_skin_condition(self, image: Any) -> Dict[str, Any]:
        """Analyze a skin condition image.

        Args:
            image: Photo of the skin area.

        Returns:
            Dict with ``raw_analysis`` (full text) and ``findings``
            (list of strings).
        """
        logger.info("Analyzing skin condition image")

        prompt = (
            "Analyze this skin image for a community health worker.\n\n"
            "Describe:\n"
            "1. APPEARANCE: Color, texture, size, distribution of any "
            "lesions/rashes\n"
            "2. LIKELY CONDITIONS: Top 3 possible diagnoses with "
            "confidence\n"
            "3. RED FLAGS: Any signs requiring immediate referral\n"
            "4. RECOMMENDED ACTIONS: Next steps for the CHW\n\n"
            "Format your response with clear sections."
        )
        response = self._run_inference(image, prompt)
        findings = self._parse_findings(response)
        logger.info("Skin analysis complete: %d findings", len(findings))
        return {
            "raw_analysis": response,
            "findings": findings,
        }

    def analyze_wound(self, image: Any) -> Dict[str, Any]:
        """Analyze a wound image.

        Args:
            image: Photo of the wound.

        Returns:
            Dict with ``raw_analysis`` (full text) and ``findings``
            (list of strings).
        """
        logger.info("Analyzing wound image")

        prompt = (
            "Analyze this wound image for a community health worker.\n\n"
            "Assess:\n"
            "1. WOUND TYPE: Cut, abrasion, burn, ulcer, etc.\n"
            "2. SIZE & DEPTH: Approximate dimensions and depth\n"
            "3. INFECTION SIGNS: Redness, swelling, pus, odor "
            "indicators\n"
            "4. HEALING STAGE: Inflammatory, proliferative, or "
            "remodeling\n"
            "5. CARE RECOMMENDATIONS: Cleaning, dressing, medications\n"
            "6. REFERRAL NEEDED: Yes/No and why\n\n"
            "Provide practical guidance for wound management."
        )
        response = self._run_inference(image, prompt)
        findings = self._parse_findings(response)
        logger.info("Wound analysis complete: %d findings", len(findings))
        return {
            "raw_analysis": response,
            "findings": findings,
        }

    def extract_medications(self, image: Any) -> List[str]:
        """Extract medication names from a prescription or label photo.

        Args:
            image: Photo of medications or prescriptions.

        Returns:
            List of medication name strings.
        """
        logger.info("Extracting medications from image")

        prompt = (
            "Extract all medication names visible in this image.\n\n"
            "This may be a photo of:\n"
            "- Prescription paper\n"
            "- Medicine bottles/boxes\n"
            "- Pill packaging\n\n"
            "List ONLY the medication names, one per line.\n"
            "Include brand names and generic names if both are visible.\n"
            "If you cannot read a name clearly, indicate [unclear].\n\n"
            "Medication names found:"
        )
        response = self._run_inference(image, prompt)
        medications = self._parse_medication_list(response)
        logger.info("Medication extraction complete: %d medications found", len(medications))
        return medications

    def extract_lab_values(self, image: Any) -> Dict[str, Any]:
        """Extract lab results from a lab report photo.

        Args:
            image: Photo of the lab report.

        Returns:
            Dict with ``raw_text`` and ``values`` (name-to-value mapping).
        """
        logger.info("Extracting lab values from image")

        prompt = (
            "Extract all lab test results from this lab report image.\n\n"
            "For each test found, provide:\n"
            "- Test name\n"
            "- Result value with units\n"
            "- Reference range if visible\n"
            "- Flag if abnormal (HIGH/LOW)\n\n"
            "Format as:\n"
            "TEST_NAME: VALUE (REFERENCE_RANGE) [FLAG if abnormal]"
        )
        response = self._run_inference(image, prompt)
        values = self._parse_lab_values(response)
        logger.info("Lab value extraction complete: %d values found", len(values))
        return {
            "raw_text": response,
            "values": values,
        }

    def _parse_findings(self, response: str) -> List[str]:
        """Parse model output into a list of finding strings.

        Args:
            response: Raw model response text.

        Returns:
            Up to 10 cleaned finding lines.
        """
        findings: List[str] = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line)
            if len(line) < 10 or line.endswith(":"):
                continue
            findings.append(line)
        return findings[:10]

    def _parse_medication_list(self, response: str) -> List[str]:
        """Parse medication names from model output.

        Args:
            response: Raw model response text.

        Returns:
            List of medication name strings.
        """
        medications: List[str] = []
        skip_patterns = [
            "medication", "found", "visible", "image", "photo",
            "prescription", "unclear", "cannot", "list",
        ]

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line)
            if (
                any(p in line.lower() for p in skip_patterns)
                and len(line) > 30
            ):
                continue
            med_name = line.strip("- \u2022*")
            if med_name and len(med_name) > 2:
                medications.append(med_name)
        return medications

    def _parse_lab_values(self, response: str) -> Dict[str, str]:
        """Parse lab test name/value pairs from model output.

        Args:
            response: Raw model response text.

        Returns:
            Mapping of test names to result strings.
        """
        values: Dict[str, str] = {}
        for line in response.split("\n"):
            match = re.match(r"([A-Za-z\s]+):\s*(.+)", line)
            if match:
                values[match.group(1).strip()] = match.group(2).strip()
        return values
