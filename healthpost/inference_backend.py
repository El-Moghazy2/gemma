"""Inference backend abstraction for MedGemma.

Provides a unified interface for text and vision inference via
**HuggingFaceBackend** using plain ``transformers`` with optional
4-bit quantization.
"""

import logging
from typing import Any, Protocol, runtime_checkable

from .config import Config

logger = logging.getLogger(__name__)


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol that all inference backends must implement."""

    @property
    def supports_vision(self) -> bool:
        """Whether this backend supports image analysis."""
        return True

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate a text response from a prompt."""
        ...

    def generate_with_image(
        self,
        image: Any,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response from an image and prompt."""
        ...


class HuggingFaceBackend:
    """Inference backend using HuggingFace transformers.

    Loads the model lazily on first call.  Supports 4-bit quantization
    on CUDA via ``bitsandbytes``.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = None
        self._processor = None

    @property
    def supports_vision(self) -> bool:
        """HuggingFace MedGemma models support vision natively."""
        return True

    def _init_model(self) -> None:
        """Load the MedGemma model if not already loaded."""
        if self._model is not None:
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch

        model_id = self.config.medgemma_model_id
        logger.info("Loading HuggingFace model: %s", model_id)

        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True,
        )

        if (
            self.config.use_4bit_quantization
            and self.config.device == "cuda"
        ):
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            dtype = (
                torch.bfloat16
                if self.config.device == "cuda"
                else torch.float32
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            self._model.to(self.config.device)

        logger.info("HuggingFace backend loaded on %s", self.config.device)

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        self._init_model()
        import torch

        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]
        logger.info(
            "HF generating: input_tokens=%d, max_new_tokens=%d, "
            "temperature=%.2f",
            input_len, max_tokens, temperature,
        )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        output_len = outputs[0].shape[-1] - input_len
        response = self._processor.decode(
            outputs[0][input_len:], skip_special_tokens=True,
        )
        logger.info(
            "HF generation complete: output_tokens=%d, response_length=%d",
            output_len, len(response),
        )
        logger.debug("HF full response: %s", response)
        return response

    def generate_with_image(
        self,
        image: Any,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        self._init_model()
        import torch

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]
        logger.info(
            "HF vision generating: input_tokens=%d, max_new_tokens=%d, "
            "temperature=%.2f",
            input_len, max_tokens, temperature,
        )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        output_len = outputs[0].shape[-1] - input_len
        response = self._processor.decode(
            outputs[0][input_len:], skip_special_tokens=True,
        )
        logger.info(
            "HF vision complete: output_tokens=%d, response_length=%d",
            output_len, len(response),
        )
        logger.debug("HF vision full response: %s", response)
        return response


def create_backend(config: Config) -> InferenceBackend:
    """Create the HuggingFace inference backend.

    Args:
        config: Application configuration.

    Returns:
        A :class:`HuggingFaceBackend` instance.
    """
    logger.info("Using HuggingFace backend: %s", config.medgemma_model_id)
    return HuggingFaceBackend(config)
