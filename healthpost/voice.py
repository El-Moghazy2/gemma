"""Voice transcription module for medical speech-to-text.

Uses Google MedASR for medical-domain speech recognition, with
OpenAI Whisper-small as a fallback.
"""

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

from .config import Config

logger = logging.getLogger(__name__)

MEDASR_MODEL_ID = "google/medasr"
WHISPER_MODEL_ID = "openai/whisper-small"


class VoiceTranscriber:
    """Medical speech-to-text transcriber.

    Attributes:
        config: Application configuration.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._pipe = None
        self._backend: str = "unknown"

    @property
    def source_label(self) -> str:
        """Human-readable label for the transcription backend."""
        if self._backend == "medasr":
            return f"Transcribed via MedASR ({MEDASR_MODEL_ID})"
        return f"Transcribed via Whisper ({WHISPER_MODEL_ID})"

    def _load_model(self) -> None:
        if self._pipe is not None:
            return

        import torch
        from transformers import pipeline

        dtype = torch.float16 if self.config.device == "cuda" else torch.float32

        # Try MedASR first
        try:
            logger.info("Loading MedASR (%s)...", self.config.medasr_model_id)
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.config.medasr_model_id,
                torch_dtype=dtype,
                device=self.config.device,
            )
            self._backend = "medasr"
            logger.info("MedASR loaded successfully")
            return
        except Exception as exc:
            logger.warning("MedASR failed to load (%s), falling back to Whisper", exc)

        # Fallback to Whisper
        logger.info("Loading Whisper (%s)...", WHISPER_MODEL_ID)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL_ID,
            torch_dtype=dtype,
            device=self.config.device,
        )
        self._backend = "whisper"
        logger.info("Whisper loaded successfully")

    def transcribe(
        self,
        audio: Union[np.ndarray, str, Path, Any],
        language: str = "en",
    ) -> str:
        """Transcribe audio to text."""
        self._load_model()
        audio_array, sample_rate = self._prepare_audio(audio)

        target_rate = self.config.sample_rate
        if sample_rate != target_rate:
            import librosa
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=target_rate,
            )
            sample_rate = target_rate

        # MedASR needs chunking params for long audio
        call_kwargs: dict[str, Any] = {}
        if self._backend == "medasr":
            call_kwargs["chunk_length_s"] = 20
            call_kwargs["stride_length_s"] = 2

        result = self._pipe(
            {"array": audio_array, "sampling_rate": sample_rate},
            **call_kwargs,
        )
        return result["text"].strip()

    def _prepare_audio(
        self, audio: Union[np.ndarray, str, Path, Any],
    ) -> tuple[np.ndarray, int]:
        """Normalize audio input to ``(array, sample_rate)``."""
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
        """Load audio samples from a file on disk."""
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
