"""
ASR/TTS Processing Node

Handles text-to-speech synthesis using Bark and automatic speech recognition using Whisper.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional
import torch
import torchaudio

from transformers import AutoProcessor, BarkModel
import whisper

logger = logging.getLogger(__name__)


class BarkWhisperProcessor:
    """Speech synthesis and recognition using Bark and Whisper"""
    
    def __init__(
        self, 
        bark_model: str = "suno/bark", 
        whisper_model: str = "large-v3",
        output_dir: Optional[str] = None
    ):
        """Initialize TTS and ASR models"""
        self.bark_model_name = bark_model
        self.whisper_model_name = whisper_model
        self.output_dir = Path(output_dir or os.getenv("OUTPUT_DIR", "./runs")) / "speech"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models lazily
        self._bark_processor = None
        self._bark_model = None
        self._whisper_model = None
        
        # Audio settings
        self.sample_rate = 24000  # Bark's default sample rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"BarkWhisperProcessor initialized with device: {self.device}")
    
    @property
    def bark_processor(self):
        """Lazy load Bark processor"""
        if self._bark_processor is None:
            try:
                self._bark_processor = AutoProcessor.from_pretrained(self.bark_model_name)
                logger.info("Bark processor loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Bark processor: {e}")
                raise
        return self._bark_processor
    
    @property 
    def bark_model(self):
        """Lazy load Bark model"""
        if self._bark_model is None:
            try:
                self._bark_model = BarkModel.from_pretrained(self.bark_model_name)
                self._bark_model = self._bark_model.to(self.device)
                logger.info("Bark model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Bark model: {e}")
                raise
        return self._bark_model
    
    @property
    def whisper_model(self):
        """Lazy load Whisper model"""
        if self._whisper_model is None:
            try:
                self._whisper_model = whisper.load_model(self.whisper_model_name, device=self.device)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
        return self._whisper_model
    
    async def process(self, text_query: str) -> Dict:
        """
        Process text through TTS and ASR pipeline
        
        Args:
            text_query: Text to synthesize and recognize
            
        Returns:
            Dict containing speech_path, asr_text, and quality scores
        """
        try:
            # Generate speech from text
            speech_path = await self._text_to_speech(text_query)
            
            # Recognize speech back to text
            asr_result = await self._speech_to_text(speech_path)
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(text_query, asr_result["text"])
            
            return {
                "speech_path": str(speech_path),
                "asr_text": asr_result["text"],
                "speech_quality_score": quality_score,
                "asr_confidence": asr_result.get("confidence", 0.0),
                "processing_metadata": {
                    "original_text": text_query,
                    "text_length": len(text_query),
                    "speech_duration": asr_result.get("duration", 0.0),
                    "language": asr_result.get("language", "en")
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process speech pipeline: {e}")
            return self._create_fallback_result(text_query)
    
    async def _text_to_speech(self, text: str) -> Path:
        """Convert text to speech using Bark"""
        try:
            # Clean and prepare text
            clean_text = self._clean_text_for_tts(text)
            
            # Process text
            inputs = self.bark_processor(clean_text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate speech
            with torch.no_grad():
                audio_array = self.bark_model.generate(**inputs, do_sample=True)
            
            # Convert to audio tensor
            audio_array = audio_array.cpu().numpy().squeeze()
            
            # Save audio file
            output_path = self.output_dir / f"tts_{hash(text) % 100000:05d}.wav"
            
            # Convert numpy array to tensor for torchaudio
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            torchaudio.save(str(output_path), audio_tensor, self.sample_rate)
            
            logger.info(f"Generated speech: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            # Create silent audio as fallback
            return self._create_silent_audio(text)
    
    async def _speech_to_text(self, audio_path: Path) -> Dict:
        """Convert speech to text using Whisper"""
        try:
            # Transcribe audio
            result = self.whisper_model.transcribe(
                str(audio_path),
                language="en",
                task="transcribe",
                verbose=False
            )
            
            return {
                "text": result["text"].strip(),
                "confidence": self._estimate_whisper_confidence(result),
                "duration": result.get("duration", 0.0),
                "language": result.get("language", "en"),
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "duration": 0.0,
                "language": "en",
                "segments": []
            }
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for better TTS synthesis"""
        # Remove excessive whitespace
        clean_text = " ".join(text.split())
        
        # Limit length for Bark (it works better with shorter texts)
        if len(clean_text) > 200:
            clean_text = clean_text[:200] + "..."
        
        # Ensure it ends with punctuation
        if clean_text and not clean_text[-1] in ".!?":
            clean_text += "."
        
        return clean_text
    
    def _estimate_whisper_confidence(self, whisper_result: Dict) -> float:
        """Estimate confidence score from Whisper result"""
        try:
            # Use average log probability from segments if available
            segments = whisper_result.get("segments", [])
            if segments:
                avg_logprob = sum(seg.get("avg_logprob", -1.0) for seg in segments) / len(segments)
                # Convert log probability to confidence (rough approximation)
                confidence = max(0.0, min(1.0, (avg_logprob + 1.0)))
                return confidence
            else:
                # Fallback based on text length and basic heuristics
                text = whisper_result.get("text", "")
                if len(text.strip()) > 0:
                    return 0.7  # Default moderate confidence
                else:
                    return 0.0
        except Exception:
            return 0.5  # Default fallback
    
    def _calculate_quality_score(self, original_text: str, transcribed_text: str) -> float:
        """Calculate quality score based on text similarity"""
        try:
            from difflib import SequenceMatcher
            
            # Normalize texts
            orig_norm = original_text.lower().strip()
            trans_norm = transcribed_text.lower().strip()
            
            # Calculate similarity
            similarity = SequenceMatcher(None, orig_norm, trans_norm).ratio()
            
            # Adjust for length differences
            length_penalty = abs(len(orig_norm) - len(trans_norm)) / max(len(orig_norm), len(trans_norm), 1)
            quality_score = similarity * (1 - length_penalty * 0.5)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.5
    
    def _create_silent_audio(self, text: str) -> Path:
        """Create silent audio file as fallback"""
        try:
            # Create 2 seconds of silence
            duration = 2.0
            silent_audio = torch.zeros(1, int(self.sample_rate * duration))
            
            output_path = self.output_dir / f"silent_{hash(text) % 100000:05d}.wav"
            torchaudio.save(str(output_path), silent_audio, self.sample_rate)
            
            logger.warning(f"Created silent audio fallback: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create silent audio: {e}")
            raise
    
    def _create_fallback_result(self, text: str) -> Dict:
        """Create fallback result when processing fails"""
        try:
            silent_path = self._create_silent_audio(text)
            return {
                "speech_path": str(silent_path),
                "asr_text": "",
                "speech_quality_score": 0.0,
                "asr_confidence": 0.0,
                "processing_metadata": {
                    "original_text": text,
                    "text_length": len(text),
                    "speech_duration": 2.0,
                    "language": "en",
                    "error": "Fallback result due to processing failure"
                }
            }
        except Exception:
            # Last resort fallback
            return {
                "speech_path": "",
                "asr_text": "",
                "speech_quality_score": 0.0,
                "asr_confidence": 0.0,
                "processing_metadata": {
                    "original_text": text,
                    "error": "Complete processing failure"
                }
            }
