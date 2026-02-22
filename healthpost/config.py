"""Configuration for the HealthPost application."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Application-wide configuration for HealthPost.

    Attributes:
        hf_model_id: Hugging Face model ID for MedGemma.
        medasr_model_id: Hugging Face model ID for MedASR.
        device: Torch device string (ZeroGPU assigns GPU dynamically).
        hf_use_4bit: Whether to use 4-bit quantization to fit in ZeroGPU VRAM.
        data_dir: Directory containing static data assets.
        max_new_tokens: Maximum tokens per generation call.
        temperature: Sampling temperature for inference.
        confidence_threshold: Minimum confidence before recommending
            referral.
        sample_rate: Expected audio sample rate in Hz.
    """

    hf_model_id: str = "google/medgemma-4b-it"
    medasr_model_id: str = "google/medasr"
    device: str = "cpu"
    hf_use_4bit: bool = True

    data_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data"
    )
    max_new_tokens: int = 512
    temperature: float = 0.3
    confidence_threshold: float = 0.7
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


default_config = Config()
