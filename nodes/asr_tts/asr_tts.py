"""
ASR/TTS Node using Bark for speech synthesis and Whisper for speech recognition

Converts text queries to speech using Bark TTS, then validates the conversion
using Whisper ASR to ensure quality.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torchaudio
import whisper
from transformers import AutoProcessor, BarkModel

# Suppress warnings from audio libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s:%(lineno)d  %(message)s"
)


class BarkWhisperProcessor:
    """
    Combined TTS (Bark) and ASR (Whisper) processor for speech-text validation.
    """

    def __init__(
        self,
        bark_model: str = "suno/bark",
        whisper_model: str = "large-v3",
        output_dir: str = "runs/current"
    ):
        """
        Initialize the Bark+Whisper processor.
        
        Args:
            bark_model: Bark model identifier
            whisper_model: Whisper model size
            output_dir: Directory for output audio files
        """
        self.bark_model_name = bark_model
        self.whisper_model_name = whisper_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create audio output directory
        self.audio_output_dir = self.output_dir / "audio"
        self.audio_output_dir.mkdir(exist_ok=True)

        # Initialize models (lazy loading)
        self.bark_processor = None
        self.bark_model = None
        self.whisper_model = None

        # Audio parameters
        self.sample_rate = 24000  # Bark's native sample rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"BarkWhisperProcessor initialized (device: {self.device})")

    def _load_bark_model(self):
        """Lazy load Bark TTS model."""
        if self.bark_model is None:
            try:
                logger.info(f"Loading Bark model: {self.bark_model_name}")

                # Load Bark processor and model
                self.bark_processor = AutoProcessor.from_pretrained(self.bark_model_name)
                self.bark_model = BarkModel.from_pretrained(
                    self.bark_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)

                logger.info("Bark model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load Bark model: {e}")
                raise

    def _load_whisper_model(self):
        """Lazy load Whisper ASR model."""
        if self.whisper_model is None:
            try:
                logger.info(f"Loading Whisper model: {self.whisper_model_name}")

                # Load Whisper model
                self.whisper_model = whisper.load_model(
                    self.whisper_model_name,
                    device=self.device
                )

                logger.info("Whisper model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise

    def synthesize_speech(self, text: str, sample_id: str) -> Dict[str, Any]:
        """
        Convert text to speech using Bark TTS.
        
        Args:
            text: Text to synthesize
            sample_id: Sample identifier for output filename
            
        Returns:
            Dict containing speech_path and synthesis metadata
        """
        try:
            # Load Bark model if needed
            self._load_bark_model()

            # Prepare text for synthesis
            # Bark works better with shorter chunks
            if len(text) > 200:
                logger.warning(f"Text is long ({len(text)} chars), truncating for better synthesis")
                text = text[:200] + "..."

            # Generate speech
            logger.info(f"Synthesizing speech for: {text[:50]}...")

            # Prepare inputs
            inputs = self.bark_processor(
                text=[text],
                return_tensors="pt",
            )

            # Move the ordinary tensors
            inputs = inputs.to(self.device)

            # Generate audio
            with torch.no_grad():
                audio_array = self.bark_model.generate(**inputs)

            # Convert to numpy and ensure the correct shape
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()

            # Bark returns shape (1, seq_len), we want (seq_len,)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.squeeze()

            # Normalize audio
            audio_array = audio_array / np.max(np.abs(audio_array))

            # Save audio file
            output_filename = f"{sample_id}_speech.wav"
            output_path = self.audio_output_dir / output_filename

            # Convert to tensor and save using torchaudio
            audio_tensor = torch.from_numpy(audio_array).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension

            torchaudio.save(
                str(output_path),
                audio_tensor,
                self.sample_rate,
                format="wav"
            )

            logger.info(f"Speech synthesized successfully: {output_path}")

            return {
                "speech_path": str(output_path),
                "duration_seconds": len(audio_array) / self.sample_rate,
                "sample_rate": self.sample_rate,
                "synthesis_successful": True
            }

        except Exception as e:
            logger.error("Speech synthesis failed: %s", e, exc_info=True)
            return {
                "speech_path": None,
                "duration_seconds": 0,
                "sample_rate": self.sample_rate,
                "synthesis_successful": False,
                "error": str(e)
            }

    def recognize_speech(self, audio_path: str) -> Dict[str, Any]:
        """
        Convert speech to text using Whisper ASR.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict containing ASR results and metadata
        """
        try:
            # Load Whisper model if needed
            self._load_whisper_model()

            logger.info(f"Recognizing speech from: {Path(audio_path).name}")

            # Transcribe audio
            transcribe_result = self.whisper_model.transcribe(
                audio_path,
                language="en",  # Assume English for medical texts
                task="transcribe"
            )

            # Extract text and confidence information
            transcribed_text = transcribe_result["text"].strip()

            # Calculate average confidence from segments
            segments = transcribe_result.get("segments", [])
            if segments:
                confidences = []
                for segment in segments:
                    # Whisper does not always provide confidence, use alternative metrics
                    if "confidence" in segment:
                        confidences.append(segment["confidence"])
                    elif "avg_logprob" in segment:
                        # Convert log probability to approximate confidence
                        confidences.append(min(1.0, max(0.0, (segment["avg_logprob"] + 1.0))))

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            else:
                avg_confidence = 0.5

            logger.info(f"ASR completed: '{transcribed_text[:50]}...'")

            return {
                "asr_text": transcribed_text,
                "confidence": avg_confidence,
                "language": transcribe_result.get("language", "en"),
                "segments": segments,
                "recognition_successful": True
            }

        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            return {
                "asr_text": "",
                "confidence": 0.0,
                "language": "unknown",
                "segments": [],
                "recognition_successful": False,
                "error": str(e)
            }

    def calculate_speech_quality(
        self,
        original_text: str,
        asr_text: str,
        synthesis_metadata: Dict[str, Any],
        asr_metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate speech quality score based on TTS-ASR round-trip.
        
        Args:
            original_text: Original input text
            asr_text: Text recognized from synthesized speech
            synthesis_metadata: TTS metadata
            asr_metadata: ASR metadata
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Check if both synthesis and recognition were successful
            if not synthesis_metadata.get("synthesis_successful", False):
                return 0.0

            if not asr_metadata.get("recognition_successful", False):
                return 0.0

            # Calculate text similarity using simple metrics
            similarity_score = self._calculate_text_similarity(original_text, asr_text)

            # Consider ASR confidence
            asr_confidence = asr_metadata.get("confidence", 0.0)

            # Consider audio duration (very short or very long might indicate issues)
            duration = synthesis_metadata.get("duration_seconds", 0)
            duration_score = min(1.0, max(0.1, duration / 10.0))  # Normalize around 10 seconds

            # Combine scores
            quality_score = (similarity_score * 0.6 + asr_confidence * 0.3 + duration_score * 0.1)

            return max(0.0, min(1.0, quality_score))

        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.0

    @staticmethod
    def _calculate_text_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using simple metrics.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Normalize texts
            text1 = text1.lower().strip()
            text2 = text2.lower().strip()

            if not text1 or not text2:
                return 0.0

            # Calculate word-level Jaccard similarity
            words1 = set(text1.split())
            words2 = set(text2.split())

            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            jaccard_similarity = intersection / union if union > 0 else 0.0

            # Calculate character-level similarity (simple)
            max_len = max(len(text1), len(text2))
            min_len = min(len(text1), len(text2))
            length_similarity = min_len / max_len if max_len > 0 else 0.0

            # Combine similarities
            overall_similarity = (jaccard_similarity * 0.7 + length_similarity * 0.3)

            return overall_similarity

        except Exception as e:
            logger.error(f"Text similarity calculation failed: {e}")
            return 0.0

    def process_text_to_speech_validation(self, text: str, sample_id: str) -> Dict[str, Any]:
        """
        Complete TTS-ASR validation pipeline.
        
        Args:
            text: Input text to process
            sample_id: Sample identifier
            
        Returns:
            Dict containing all outputs and quality metrics
        """
        logger.info(f"Starting TTS-ASR processing for sample: {sample_id}")

        try:
            # Step 1: Synthesize speech
            synthesis_result = self.synthesize_speech(text, sample_id)

            # Step 2: Recognize speech (if synthesis was successful)
            if synthesis_result["synthesis_successful"] and synthesis_result["speech_path"]:
                asr_result = self.recognize_speech(synthesis_result["speech_path"])
            else:
                asr_result = {
                    "asr_text": "",
                    "confidence": 0.0,
                    "recognition_successful": False,
                    "error": "No audio to recognize"
                }

            # Step 3: Calculate quality score
            quality_score = self.calculate_speech_quality(
                text,
                asr_result["asr_text"],
                synthesis_result,
                asr_result
            )

            # Combine results
            combine_result = {
                "speech_path": synthesis_result["speech_path"],
                "asr_text": asr_result["asr_text"],
                "speech_quality_score": quality_score,
                "original_text": text,
                "synthesis_metadata": synthesis_result,
                "asr_metadata": asr_result,
                "processing_successful": synthesis_result["synthesis_successful"] and
                                         asr_result["recognition_successful"]
            }

            logger.info(f"TTS-ASR processing completed (quality: {quality_score:.2f})")
            return combine_result

        except Exception as e:
            logger.error(f"TTS-ASR processing failed: {e}")
            return {
                "speech_path": None,
                "asr_text": "",
                "speech_quality_score": 0.0,
                "original_text": text,
                "processing_successful": False,
                "error": str(e)
            }


def run_asr_tts(text_query: str, sample_id: str = "sample", output_dir: str = "runs/current") -> Dict[str, Any]:
    """
    Run the Bark+Whisper ASR/TTS node.
    
    Args:
        text_query: Text to convert to speech and validate
        sample_id: Sample identifier for output files
        output_dir: Directory for output files
        
    Returns:
        Dict containing speech_path, asr_text, and speech_quality_score
        
    Raises:
        Exception: If processing fails critically
    """
    logger.info(f"Starting ASR/TTS processing for: {text_query[:50]}...")

    try:
        processor = BarkWhisperProcessor(output_dir=output_dir)
        validation_result = processor.process_text_to_speech_validation(text_query, sample_id)

        # Add node metadata
        validation_result["processor"] = "BarkWhisperProcessor"
        validation_result["processor_version"] = "v1.0.0"
        validation_result["tts_model"] = "suno/bark"
        validation_result["asr_model"] = "whisper-large-v3"

        if validation_result["processing_successful"]:
            logger.info("ASR/TTS processing completed successfully")
        else:
            logger.warning("ASR/TTS processing completed with issues")

        return validation_result

    except Exception as e:
        logger.error(f"ASR/TTS processing failed: {e}")
        # Return a fallback result
        return {
            "speech_path": None,
            "asr_text": "",
            "speech_quality_score": 0.0,
            "processor": "BarkWhisperProcessor",
            "processor_version": "v1.0.0",
            "processing_successful": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the ASR/TTS processor
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_text = "What abnormality is visible in this chest X-ray image?"

        result = run_asr_tts(
            text_query=test_text,
            sample_id="test_001",
            output_dir=temp_dir
        )

        print("ASR/TTS test results:")
        print(f"  Original text: {test_text}")
        print(f"  ASR text: {result['asr_text']}")
        print(f"  Quality score: {result['speech_quality_score']:.2f}")
        print(f"  Speech file: {result['speech_path']}")
        print(f"  Processing successful: {result['processing_successful']}")

        if result['speech_path'] and Path(result['speech_path']).exists():
            print(f"  Audio file size: {Path(result['speech_path']).stat().st_size} bytes")

        if 'error' in result:
            print(f"  Error: {result['error']}")
