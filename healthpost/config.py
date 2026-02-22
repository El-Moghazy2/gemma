"""Configuration for the HealthPost application."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def detect_device() -> str:
    """Return the target device string.

    On HF Spaces with ZeroGPU, CUDA must NOT be initialised in the
    main process — ``@spaces.GPU`` handles GPU assignment at call time.
    We therefore always return ``'cpu'`` on Spaces so that models load
    on CPU; ZeroGPU transparently moves them to GPU when needed.

    For local runs we probe ``torch.cuda.is_available()``.
    """
    if os.environ.get("SPACE_ID"):
        # ZeroGPU: load on CPU, GPU is assigned dynamically
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@dataclass
class Config:
    """Application-wide configuration for HealthPost.

    Attributes:
        hf_model_id: Hugging Face model ID for MedGemma.
        medasr_model_id: Hugging Face model ID for MedASR speech-to-text.
        device: Torch device string (auto-detected; ``'cuda'`` on Spaces/GPU).
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
    device: str = field(default_factory=detect_device)
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
