"""
Voice transcription module using MedASR for medical speech-to-text.

Handles symptom capture from patient or CHW voice descriptions.
"""

from typing import Union, Optional, Any
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VoiceTranscriber:
    """
    Medical speech-to-text transcription using MedASR.

    Optimized for medical terminology and symptom descriptions.
    Falls back to Whisper if MedASR is not available.
    """

    def __init__(self, config):
        """Initialize the voice transcriber."""
        self.config = config
        self._model = None
        self._processor = None
        self._model_type = None  # "medasr", "whisper", or "mock"

    def _load_model(self):
        """Lazy load the ASR model."""
        if self._model is not None:
            return

        # Try MedASR first
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import torch

            logger.info("Loading MedASR model...")
            self._processor = AutoProcessor.from_pretrained(
                self.config.medasr_model_id,
                trust_remote_code=True,
            )
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.medasr_model_id,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                trust_remote_code=True,
            )
            self._model.to(self.config.device)
            self._model_type = "medasr"
            logger.info("MedASR model loaded successfully")
            return
        except Exception as e:
            logger.warning(f"Failed to load MedASR: {e}")

        # Fall back to Whisper
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            import torch

            logger.info("Falling back to Whisper for ASR...")
            model_id = "openai/whisper-small"
            self._processor = WhisperProcessor.from_pretrained(model_id)
            self._model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            )
            self._model.to(self.config.device)
            self._model_type = "whisper"
            logger.info("Whisper model loaded successfully")
            return
        except Exception as e:
            logger.warning(f"Failed to load Whisper: {e}")

        # Mock mode for testing without models
        logger.warning("No ASR model available, using mock mode")
        self._model_type = "mock"
        self._model = "mock"

    def transcribe(
        self,
        audio: Union[np.ndarray, str, Path, Any],
        language: str = "en",
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio input - can be:
                - numpy array of audio samples
                - path to audio file
                - Gradio audio tuple (sample_rate, data)
            language: Language code (default: "en")

        Returns:
            Transcribed text
        """
        self._load_model()

        # Handle different input types
        audio_array, sample_rate = self._prepare_audio(audio)

        if self._model_type == "mock":
            return self._mock_transcribe(audio_array)

        try:
            import torch

            # Process audio
            inputs = self._processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Generate transcription
            with torch.no_grad():
                if self._model_type == "whisper":
                    forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                        language=language, task="transcribe"
                    )
                    generated_ids = self._model.generate(
                        inputs["input_features"],
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=256,
                    )
                else:
                    # MedASR
                    generated_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=256,
                    )

            # Decode
            transcription = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return transcription.strip()

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return f"[Transcription error: {e}]"

    def _prepare_audio(
        self, audio: Union[np.ndarray, str, Path, Any]
    ) -> tuple[np.ndarray, int]:
        """Prepare audio input for processing."""

        # Gradio audio tuple: (sample_rate, data)
        if isinstance(audio, tuple) and len(audio) == 2:
            sample_rate, audio_array = audio
            if isinstance(audio_array, np.ndarray):
                # Convert to float and normalize
                if audio_array.dtype in [np.int16, np.int32]:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                # Convert stereo to mono
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                return audio_array, sample_rate

        # Numpy array
        if isinstance(audio, np.ndarray):
            return audio, self.config.sample_rate

        # File path
        if isinstance(audio, (str, Path)):
            return self._load_audio_file(audio)

        # Unknown type - try to extract data
        try:
            if hasattr(audio, "numpy"):
                return audio.numpy(), self.config.sample_rate
        except Exception:
            pass

        raise ValueError(f"Unsupported audio type: {type(audio)}")

    def _load_audio_file(self, path: Union[str, Path]) -> tuple[np.ndarray, int]:
        """Load audio from file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Try soundfile first
        try:
            import soundfile as sf
            audio_array, sample_rate = sf.read(path)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array.astype(np.float32), sample_rate
        except ImportError:
            pass

        # Try librosa
        try:
            import librosa
            audio_array, sample_rate = librosa.load(path, sr=self.config.sample_rate)
            return audio_array, sample_rate
        except ImportError:
            pass

        # Try scipy
        try:
            from scipy.io import wavfile
            sample_rate, audio_array = wavfile.read(path)
            audio_array = audio_array.astype(np.float32) / 32768.0
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array, sample_rate
        except ImportError:
            pass

        raise ImportError("No audio loading library available (soundfile, librosa, scipy)")

    def _mock_transcribe(self, audio: np.ndarray) -> str:
        """Mock transcription for testing."""
        # Return a sample symptom description
        duration = len(audio) / self.config.sample_rate if len(audio) > 0 else 0

        if duration < 1:
            return "Patient reports feeling unwell."
        elif duration < 5:
            return "Patient has fever and headache for two days."
        else:
            return (
                "Patient presents with high fever for three days, "
                "accompanied by headache, body aches, and a rash on the trunk. "
                "No cough or respiratory symptoms."
            )
