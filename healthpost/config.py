"""Configuration for the HealthPost application."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Application-wide configuration for HealthPost.

    Attributes:
        ollama_model: Ollama model tag for MedGemma.
        data_dir: Directory containing static data assets.
        max_new_tokens: Maximum tokens per generation call.
        temperature: Sampling temperature for inference.
        confidence_threshold: Minimum confidence before recommending
            referral.
        sample_rate: Expected audio sample rate in Hz.
    """

    ollama_model: str = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:latest"

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
