"""Inference backend abstraction for MedGemma.

Provides a unified interface for text, vision, and structured inference.

Backends:
- **TransformersBackend** (default) -- uses HF ``transformers`` + ``AutoProcessor``.
- **OllamaBackend** -- uses a local Ollama server (kept for local dev fallback).
"""

import json
import logging
import re
from io import BytesIO
from typing import Any, List, Optional, Protocol, Type, runtime_checkable

from pydantic import BaseModel

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

    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate a JSON response constrained to a Pydantic schema.

        Returns the raw JSON string; the caller validates with Pydantic.
        """
        ...

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response from a multi-turn conversation.

        Args:
            messages: Full conversation (system + history + new user message).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's response text.
        """
        ...


# ---------------------------------------------------------------------------
# TransformersBackend
# ---------------------------------------------------------------------------

class TransformersBackend:
    """Inference backend using HF ``transformers``.

    Loads MedGemma via ``AutoModelForCausalLM`` + ``AutoProcessor`` with
    optional 4-bit quantization for ZeroGPU VRAM constraints.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the model and processor on first call."""
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = self.config.hf_model_id
        logger.info("Loading model %s ...", model_id)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
        }

        if self.config.hf_use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info("4-bit quantization enabled")
            except ImportError:
                logger.warning(
                    "bitsandbytes not available; loading without quantization"
                )

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, **load_kwargs
        )
        self._loaded = True
        logger.info("Model loaded successfully")

    # -- core generation helper ---------------------------------------------

    def _generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 512,
        images: Optional[List[Any]] = None,
    ) -> str:
        """Shared generation logic for all public methods."""
        import torch

        self._ensure_loaded()

        processor_kwargs: dict[str, Any] = {
            "text": self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ),
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images

        device = next(self._model.parameters()).device
        inputs = self._processor(**processor_kwargs).to(device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
            )

        new_tokens = output_ids[0][input_len:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)

        logger.info(
            "Generated: input_tokens=%d, output_tokens=%d",
            input_len, len(new_tokens),
        )
        logger.debug("Response: %s", text)
        return text

    # -- public protocol methods --------------------------------------------

    @property
    def supports_vision(self) -> bool:
        return True

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate a text response from a prompt.

        Args:
            prompt: Input text prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """
        messages = [{"role": "user", "content": prompt}]
        return self._generate(messages, temperature, max_tokens)

    def generate_with_image(
        self,
        image: Any,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response from an image and prompt.

        Args:
            image: PIL Image to analyze.
            prompt: Text prompt accompanying the image.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.

        Raises:
            ValueError: If *image* is not a PIL Image.
        """
        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            raise ValueError(
                f"Expected PIL Image, got {type(image).__name__}"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(
            messages, temperature, max_tokens, images=[image]
        )

    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate JSON constrained to *schema* via prompt engineering.

        Appends the JSON schema to the prompt and extracts the JSON object
        from the response. The caller is responsible for calling
        ``schema.model_validate_json(result)``.
        """
        json_schema = json.dumps(schema.model_json_schema(), indent=2)
        structured_prompt = (
            f"{prompt}\n\n"
            f"You MUST output ONLY valid JSON matching this schema:\n"
            f"```json\n{json_schema}\n```\n\n"
            f"Output ONLY the JSON object, no other text."
        )

        messages = [{"role": "user", "content": structured_prompt}]
        raw = self._generate(messages, temperature, max_tokens)
        return self._extract_json(raw)

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response from a multi-turn conversation.

        Args:
            messages: Full conversation including system, history, and
                user messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's response text.
        """
        return self._generate(messages, temperature, max_tokens)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract a JSON object from model output that may contain markdown fences."""
        # Try to find JSON inside markdown code fences
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
        )
        if fence_match:
            return fence_match.group(1)

        # Try to find a raw JSON object
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)

        # Return as-is and let Pydantic validation handle it
        return text.strip()


# ---------------------------------------------------------------------------
# OllamaBackend (kept for local dev fallback)
# ---------------------------------------------------------------------------

class OllamaBackend:
    """Inference backend using Ollama.

    Calls the local Ollama server via the ``ollama`` Python package.
    Supports text, vision, and structured (schema-constrained) output.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = config.hf_model_id

    @property
    def supports_vision(self) -> bool:
        return True

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate a text response from a prompt via Ollama.

        Args:
            prompt: Input text prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """
        import ollama

        logger.info(
            "Ollama generating: model=%s, max_tokens=%d, temperature=%.2f",
            self._model, max_tokens, temperature,
        )

        response = ollama.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": max_tokens, "temperature": temperature},
        )

        text = response.message.content
        tokens_in = response.prompt_eval_count or 0
        tokens_out = response.eval_count or 0
        logger.info(
            "Ollama complete: input_tokens=%d, output_tokens=%d",
            tokens_in, tokens_out,
        )
        logger.debug("Ollama response: %s", text)
        return text

    def generate_with_image(
        self,
        image: Any,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response from an image and prompt via Ollama.

        Args:
            image: PIL Image to analyze.
            prompt: Text prompt accompanying the image.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.

        Raises:
            ValueError: If *image* is not a PIL Image.
        """
        import base64

        import ollama
        from PIL import Image as PILImage

        # Encode PIL image to base64
        if isinstance(image, PILImage.Image):
            buf = BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
        else:
            raise ValueError(
                f"Expected PIL Image, got {type(image).__name__}"
            )

        logger.info(
            "Ollama vision generating: model=%s, max_tokens=%d, temperature=%.2f",
            self._model, max_tokens, temperature,
        )

        response = ollama.chat(
            model=self._model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }],
            options={"num_predict": max_tokens, "temperature": temperature},
        )

        text = response.message.content
        tokens_in = response.prompt_eval_count or 0
        tokens_out = response.eval_count or 0
        logger.info(
            "Ollama vision complete: input_tokens=%d, output_tokens=%d",
            tokens_in, tokens_out,
        )
        logger.debug("Ollama vision response: %s", text)
        return text

    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate JSON constrained to *schema* via Ollama's ``format`` parameter.

        Returns the raw JSON string. The caller is responsible for
        calling ``schema.model_validate_json(result)``.
        """
        import ollama

        logger.info(
            "Ollama structured generating: model=%s, schema=%s, "
            "max_tokens=%d, temperature=%.2f",
            self._model, schema.__name__, max_tokens, temperature,
        )

        response = ollama.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema(),
            options={"num_predict": max_tokens, "temperature": temperature},
        )

        text = response.message.content
        tokens_in = response.prompt_eval_count or 0
        tokens_out = response.eval_count or 0
        logger.info(
            "Ollama structured complete: input_tokens=%d, output_tokens=%d",
            tokens_in, tokens_out,
        )
        logger.debug("Ollama structured response: %s", text)
        return text


    def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response from a multi-turn conversation via Ollama.

        Args:
            messages: Full conversation including system, history, and
                user messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's response text.
        """
        import ollama

        logger.info(
            "Ollama chat generating: model=%s, messages=%d, "
            "max_tokens=%d, temperature=%.2f",
            self._model, len(messages), max_tokens, temperature,
        )

        response = ollama.chat(
            model=self._model,
            messages=messages,
            options={"num_predict": max_tokens, "temperature": temperature},
        )

        text = response.message.content
        tokens_in = response.prompt_eval_count or 0
        tokens_out = response.eval_count or 0
        logger.info(
            "Ollama chat complete: input_tokens=%d, output_tokens=%d",
            tokens_in, tokens_out,
        )
        logger.debug("Ollama chat response: %s", text)
        return text


def create_backend(config: Config) -> InferenceBackend:
    """Create the default inference backend.

    Returns a :class:`TransformersBackend` instance. Switch to
    :class:`OllamaBackend` for local development if needed.
    """
    logger.info("Using Transformers backend: %s", config.hf_model_id)
    return TransformersBackend(config)
