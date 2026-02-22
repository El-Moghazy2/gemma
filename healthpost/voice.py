"""Voice transcription module using MedASR for medical speech-to-text.

Handles symptom capture from patient or CHW voice descriptions.
Requires MedASR — fails if the model cannot be loaded.
"""

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


class VoiceTranscriber:
    """Medical speech-to-text transcriber.

    Optimized for medical terminology and symptom descriptions.

    Attributes:
        config: Application configuration.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the voice transcriber.

        Args:
            config: Application configuration instance.
        """
        self.config = config
        self._model = None
        self._processor = None

    @property
    def source_label(self) -> str:
        """Human-readable label for the transcription backend."""
        return "Transcribed via MedASR (google/medasr)"

    def _load_model(self) -> None:
        """Lazy-load the MedASR model.

        Raises:
            RuntimeError: If MedASR cannot be loaded.
        """
        if self._model is not None:
            return

        from transformers import AutoModel, AutoProcessor
        import torch

        logger.info("Loading MedASR model...")
        self._processor = AutoProcessor.from_pretrained(
            self.config.medasr_model_id, trust_remote_code=True,
        )
        dtype = (
            torch.float16
            if self.config.device == "cuda"
            else torch.float32
        )
        self._model = AutoModel.from_pretrained(
            self.config.medasr_model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._model.to(self.config.device)
        logger.info("MedASR model loaded successfully")

    def transcribe(
        self,
        audio: Union[np.ndarray, str, Path, Any],
        language: str = "en",
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio input as a numpy array, file path, or Gradio
                audio tuple ``(sample_rate, data)``.
            language: Language code for transcription.

        Returns:
            Transcribed text string.
        """
        self._load_model()
        audio_array, sample_rate = self._prepare_audio(audio)

        target_rate = self.config.sample_rate
        if sample_rate != target_rate:
            import librosa
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=target_rate,
            )
            sample_rate = target_rate

        import torch

        inputs = self._processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self.config.device) for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=256,
            )

        transcription = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()

    def _prepare_audio(
        self, audio: Union[np.ndarray, str, Path, Any],
    ) -> tuple[np.ndarray, int]:
        """Normalize audio input to ``(array, sample_rate)``.

        Args:
            audio: Raw audio in any supported format.

        Returns:
            Tuple of ``(float32 mono array, sample_rate)``.

        Raises:
            ValueError: If the audio type is not supported.
        """
        if isinstance(audio, tuple) and len(audio) == 2:
            sample_rate, audio_array = audio
            if isinstance(audio_array, np.ndarray):
                if audio_array.dtype in [np.int16, np.int32]:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                return audio_array, sample_rate

        if isinstance(audio, np.ndarray):
            return audio, self.config.sample_rate

        if isinstance(audio, (str, Path)):
            return self._load_audio_file(audio)

        try:
            if hasattr(audio, "numpy"):
                return audio.numpy(), self.config.sample_rate
        except Exception:
            pass

        raise ValueError(f"Unsupported audio type: {type(audio)}")

    def _load_audio_file(
        self, path: Union[str, Path],
    ) -> tuple[np.ndarray, int]:
        """Load audio samples from a file on disk.

        Args:
            path: Path to the audio file.

        Returns:
            Tuple of ``(float32 mono array, sample_rate)``.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ImportError: If no audio loading library is available.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            import soundfile as sf
            audio_array, sample_rate = sf.read(path)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array.astype(np.float32), sample_rate
        except ImportError:
            pass

        try:
            import librosa
            audio_array, sample_rate = librosa.load(
                path, sr=self.config.sample_rate,
            )
            return audio_array, sample_rate
        except ImportError:
            pass

        try:
            from scipy.io import wavfile
            sample_rate, audio_array = wavfile.read(path)
            audio_array = audio_array.astype(np.float32) / 32768.0
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array, sample_rate
        except ImportError:
            pass

        raise ImportError(
            "No audio loading library available "
            "(soundfile, librosa, or scipy)"
        )
