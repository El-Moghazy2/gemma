"""
Ollama client for local LLM inference.

Provides text and vision capabilities using locally-running Ollama models.
"""

import json
import base64
import logging
from typing import Optional, List, Any
from pathlib import Path
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for Ollama API.

    Supports both text generation and vision models (like LLaVA).
    """

    def __init__(self, host: str = "http://localhost:11434"):
        """Initialize Ollama client."""
        self.host = host.rstrip("/")
        self._available_models: Optional[List[str]] = None

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available models."""
        if self._available_models is not None:
            return self._available_models

        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                self._available_models = [m["name"] for m in data.get("models", [])]
                return self._available_models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def has_model(self, model_name: str) -> bool:
        """Check if a model is available."""
        models = self.list_models()
        # Check for exact match or prefix match (e.g., "gemma2" matches "gemma2:latest")
        return any(m == model_name or m.startswith(f"{model_name}:") for m in models)

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            data = json.dumps({"name": model_name}).encode()
            req = urllib.request.Request(
                f"{self.host}/api/pull",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            logger.info(f"Pulling model {model_name}...")
            with urllib.request.urlopen(req, timeout=600) as response:
                # Stream the response to show progress
                for line in response:
                    status = json.loads(line.decode())
                    if "status" in status:
                        print(f"  {status['status']}", end="\r")
                print()

            self._available_models = None  # Reset cache
            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        images: Optional[List[Any]] = None,
    ) -> str:
        """
        Generate text using Ollama.

        Args:
            model: Model name (e.g., "gemma2", "llava")
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            images: List of images for vision models (PIL Images, numpy arrays, or base64 strings)

        Returns:
            Generated text
        """
        # Build request
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            request_data["system"] = system

        # Handle images for vision models
        if images:
            request_data["images"] = [self._encode_image(img) for img in images]

        try:
            data = json.dumps(request_data).encode()
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode())
                return result.get("response", "")

        except urllib.error.URLError as e:
            logger.error(f"Ollama request failed: {e}")
            raise RuntimeError(f"Ollama request failed: {e}")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def chat(
        self,
        model: str,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 512,
        images: Optional[List[Any]] = None,
    ) -> str:
        """
        Chat completion using Ollama.

        Args:
            model: Model name
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            images: List of images for vision models

        Returns:
            Assistant response
        """
        # Add images to the last user message if provided
        if images and messages:
            last_msg = messages[-1].copy()
            if last_msg.get("role") == "user":
                last_msg["images"] = [self._encode_image(img) for img in images]
                messages = messages[:-1] + [last_msg]

        request_data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            data = json.dumps(request_data).encode()
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode())
                return result.get("message", {}).get("content", "")

        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise

    def _encode_image(self, image: Any) -> str:
        """Encode image to base64 string."""
        # Already base64 string
        if isinstance(image, str):
            # Check if it's a file path
            if Path(image).exists():
                with open(image, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            # Assume it's already base64
            return image

        # PIL Image
        try:
            from PIL import Image
            import io

            if isinstance(image, Image.Image):
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode()
        except ImportError:
            pass

        # Numpy array
        try:
            import numpy as np
            from PIL import Image
            import io

            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode()
        except ImportError:
            pass

        raise ValueError(f"Cannot encode image of type {type(image)}")


# Medical system prompts
MEDICAL_SYSTEM_PROMPT = """You are a medical AI assistant helping Community Health Workers (CHWs) in low-resource settings.

Your role is to:
1. Help analyze patient symptoms and suggest possible diagnoses
2. Recommend appropriate treatments that can be provided at health post level
3. Identify cases that need referral to hospital
4. Provide clear, actionable guidance

Important guidelines:
- Be concise and practical
- Focus on common conditions in community health settings
- Always consider patient safety first
- Recommend referral when diagnosis is uncertain or condition is serious
- Use simple language that CHWs can understand

Remember: You are a decision SUPPORT tool, not a replacement for clinical judgment."""

VISION_SYSTEM_PROMPT = """You are a medical image analysis assistant helping Community Health Workers.

When analyzing medical images:
1. Describe what you observe objectively
2. Suggest possible conditions this could indicate
3. Note any concerning features
4. Recommend next steps

Focus on conditions common in community health settings:
- Skin infections and rashes
- Wounds and burns
- Eye conditions
- Signs of systemic illness

Be specific but avoid overly technical language."""
