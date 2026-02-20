"""Inference backend abstraction for MedGemma.

Provides a unified interface for text, vision, and structured inference
via **OllamaBackend** using the ``ollama`` Python client.
"""

import base64
import logging
from io import BytesIO
from typing import Any, Protocol, Type, runtime_checkable

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


class OllamaBackend:
    """Inference backend using Ollama.

    Calls the local Ollama server via the ``ollama`` Python package.
    Supports text, vision, and structured (schema-constrained) output.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = config.ollama_model

    @property
    def supports_vision(self) -> bool:
        return True

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
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
    """Create the Ollama inference backend.

    Args:
        config: Application configuration.

    Returns:
        An :class:`OllamaBackend` instance.
    """
    logger.info("Using Ollama backend: %s", config.ollama_model)
    return OllamaBackend(config)
