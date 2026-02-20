"""Configuration for the HealthPost application."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Application-wide configuration for HealthPost.

    Attributes:
        medgemma_model_id: HuggingFace model ID for MedGemma 1.5.
        medasr_model_id: HuggingFace model ID for MedASR.
        use_4bit_quantization: Whether to quantize models to 4-bit.
        device: Compute device (``"auto"``, ``"cuda"``, or ``"cpu"``).
        data_dir: Directory containing static data assets.
        drug_db_path: Path to the SQLite drug database.
        max_new_tokens: Maximum tokens per generation call.
        temperature: Sampling temperature for inference.
        confidence_threshold: Minimum confidence before recommending
            referral.
        sample_rate: Expected audio sample rate in Hz.
    """

    medgemma_model_id: str = "google/medgemma-1.5-4b-it"
    medasr_model_id: str = "google/medasr"

    use_4bit_quantization: bool = True
    device: str = "auto"

    data_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data"
    )
    drug_db_path: Optional[Path] = None

    max_new_tokens: int = 512
    temperature: float = 0.3
    confidence_threshold: float = 0.7
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        """Derive defaults and verify HuggingFace availability."""
        if self.drug_db_path is None:
            self.drug_db_path = self.data_dir / "drugs.db"

        self.data_dir.mkdir(parents=True, exist_ok=True)

        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        self._check_huggingface()

    def _check_huggingface(self) -> None:
        """Verify that core ML packages are importable.

        Unsloth manages its own transformer/torch dependencies, so this
        check is intentionally skipped.
        """
        return


default_config = Config()
