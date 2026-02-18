"""
Configuration for HealthPost application.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import os


@dataclass
class Config:
    """Configuration settings for HealthPost."""

    # Backend selection: "ollama", "huggingface", or "mock"
    backend: str = "ollama"

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3n"  # For text/reasoning (or gemma2, llama3, etc.)
    ollama_vision_model: str = "llava"  # For image analysis (llava, llava-llama3, etc.)

    # HuggingFace model settings (fallback)
    medgemma_model_id: str = "google/medgemma-4b-it"
    medasr_model_id: str = "google/medasr"

    # Quantization for edge deployment
    use_4bit_quantization: bool = True

    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Database paths
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    drug_db_path: Optional[Path] = None

    # Inference settings
    max_new_tokens: int = 512
    temperature: float = 0.3  # Lower for more consistent medical advice

    # Safety thresholds
    confidence_threshold: float = 0.7  # Below this, recommend referral

    # Audio settings
    sample_rate: int = 16000

    def __post_init__(self):
        """Initialize derived settings."""
        if self.drug_db_path is None:
            self.drug_db_path = self.data_dir / "drugs.db"

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        # Check if Ollama is available
        if self.backend == "ollama":
            if not self._check_ollama():
                print("Warning: Ollama not available, falling back to mock mode")
                self.backend = "mock"

    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.ollama_host}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False


# Global default configuration
default_config = Config()
