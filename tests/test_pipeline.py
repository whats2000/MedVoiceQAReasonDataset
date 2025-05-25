"""
Basic test suite for the MedVoiceQAReasonDataset pipeline.
Tests individual nodes and full pipeline integration.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import pipeline components
from pipeline.run_pipeline import MedVoiceQAPipeline, PipelineState
from nodes.loader import VQARADLoader
from nodes.segmentation import GeminiVisionSegmenter
from nodes.asr_tts import BarkWhisperProcessor
from nodes.explanation import GeminiReasoningEngine
from nodes.validation import GeminiValidationDuo
from nodes.human_review import HumanReviewNode


class TestVQARADLoader:
    """Test the VQA-RAD data loader node."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "vqarad_data_path": "test_data/vqarad",
            "output_dir": "test_output"
        }
        self.loader = VQARADLoader(self.config)
    
    @pytest.mark.asyncio
    async def test_process_with_valid_sample(self):
        """Test processing a valid sample ID."""
        # Mock the sample data
        with patch.object(self.loader, '_load_sample_metadata') as mock_load:
            mock_load.return_value = {
                "image_path": "test.dcm",
                "question": "What is shown in this image?",
                "answer": "Test answer"
            }
            
            with patch.object(self.loader, '_convert_dicom_to_png') as mock_convert:
                mock_convert.return_value = "test_output/test.png"
                
                result = await self.loader.process({"sample_id": "test_001"})
                
                assert "image_path" in result
                assert "text_query" in result
                assert "metadata" in result
                assert result["text_query"] == "What is shown in this image?"
    
    @pytest.mark.asyncio
    async def test_process_with_invalid_sample(self):
        """Test processing an invalid sample ID."""
        with patch.object(self.loader, '_load_sample_metadata') as mock_load:
            mock_load.side_effect = FileNotFoundError("Sample not found")
            
            result = await self.loader.process({"sample_id": "invalid_001"})
            
            # Should return fallback data
            assert "placeholder" in result["image_path"]
            assert "error_loading" in result["text_query"]
    
    def test_dicom_conversion(self):
        """Test DICOM to PNG conversion logic."""
        # Test the windowing and normalization logic
        import numpy as np
        from PIL import Image
        
        # Create mock DICOM data
        mock_pixel_array = np.random.randint(0, 4096, (512, 512)).astype(np.int16)
        
        # Test the conversion function
        result = self.loader._apply_window_normalization(mock_pixel_array, 400, 1000)
        
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255


class TestGeminiVisionSegmenter:
    """Test the Gemini vision segmentation node."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "gemini_api_key": "test_key"
        }
        self.segmenter = GeminiVisionSegmenter(self.config)
    
    @pytest.mark.asyncio
    async def test_process_with_valid_image(self):
        """Test segmentation with valid image."""
        state = {
            "image_path": "test_image.png",
            "text_query": "Locate the heart in this chest X-ray"
        }
        
        # Mock the Gemini API response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "bounding_boxes": [
                {
                    "label": "heart",
                    "x": 100,
                    "y": 150,
                    "width": 200,
                    "height": 180,
                    "confidence": 0.92
                }
            ],
            "key_regions": ["cardiac silhouette", "mediastinum"]
        })
        
        with patch.object(self.segmenter.model, 'generate_content', return_value=mock_response):
            with patch('PIL.Image.open'):
                result = await self.segmenter.process(state)
                
                assert "visual_box" in result
                assert "bounding_boxes" in result["visual_box"]
                assert len(result["visual_box"]["bounding_boxes"]) == 1
                assert result["visual_box"]["bounding_boxes"][0]["label"] == "heart"
    
    @pytest.mark.asyncio
    async def test_process_with_missing_image(self):
        """Test segmentation with missing image file."""
        state = {
            "image_path": "nonexistent.png",
            "text_query": "Find something"
        }
        
        result = await self.segmenter.process(state)
        
        # Should return fallback
        assert result["visual_box"]["fallback_used"] is True
        assert "processing_failed" in result["visual_box"]["error"]


class TestBarkWhisperProcessor:
    """Test the Bark/Whisper audio processing node."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "output_dir": "test_output",
            "bark_model": "suno/bark-small",
            "whisper_model": "openai/whisper-large-v3"
        }
        self.processor = BarkWhisperProcessor(self.config)
    
    @pytest.mark.asyncio
    async def test_text_to_speech_processing(self):
        """Test text-to-speech generation."""
        # Mock the Bark model
        with patch('transformers.AutoProcessor.from_pretrained') as mock_processor:
            with patch('transformers.AutoModel.from_pretrained') as mock_model:
                with patch('scipy.io.wavfile.write') as mock_write:
                    mock_processor.return_value = Mock()
                    mock_model.return_value = Mock()
                    mock_model.return_value.generate.return_value = [Mock()]
                    
                    result = await self.processor._generate_speech("Hello, this is a test.")
                    
                    assert result.endswith('.wav')
                    mock_write.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_speech_to_text_processing(self):
        """Test speech-to-text recognition."""
        # Mock the Whisper model
        with patch('transformers.AutoModelForSpeechSeq2Seq.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                with patch('transformers.AutoFeatureExtractor.from_pretrained') as mock_extractor:
                    with patch('librosa.load') as mock_load:
                        mock_load.return_value = (Mock(), 16000)
                        mock_model.return_value = Mock()
                        mock_tokenizer.return_value = Mock()
                        mock_extractor.return_value = Mock()
                        
                        # Mock pipeline
                        mock_pipeline = Mock()
                        mock_pipeline.return_value = {"text": "Hello, this is a test."}
                        
                        with patch('transformers.pipeline', return_value=mock_pipeline):
                            result = await self.processor._transcribe_speech("test.wav")
                            
                            assert result == "Hello, this is a test."


class TestGeminiReasoningEngine:
    """Test the Gemini reasoning engine node."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "gemini_api_key": "test_key"
        }
        self.engine = GeminiReasoningEngine(self.config)
    
    @pytest.mark.asyncio
    async def test_reasoning_generation(self):
        """Test medical reasoning generation."""
        state = {
            "image_path": "test_xray.png",
            "text_query": "What abnormalities are visible?",
            "visual_box": {
                "bounding_boxes": [{"label": "opacity", "x": 100, "y": 100, "width": 50, "height": 50}]
            }
        }
        
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "reasoning_steps": [
                "Step 1: Analyzed image for abnormalities",
                "Step 2: Identified opacity in right lung field"
            ],
            "visual_evidence": "Dense opacity visible in right lower lobe",
            "medical_context": "Consistent with pneumonia or consolidation",
            "confidence_assessment": {
                "overall_confidence": 0.85,
                "reasoning": "Clear visual findings support diagnosis",
                "uncertainty_factors": ["Patient history unknown"]
            },
            "final_explanation": "The image shows a dense opacity in the right lower lobe consistent with pneumonia."
        })
        
        with patch.object(self.engine.model, 'generate_content', return_value=mock_response):
            with patch('PIL.Image.open'):
                result = await self.engine.process(state)
                
                assert "text_explanation" in result
                assert "uncertainty" in result
                assert result["uncertainty"] == 0.85
                assert "pneumonia" in result["text_explanation"]


class TestPipelineIntegration:
    """Test full pipeline integration."""
    
    def setup_method(self):
        """Setup test pipeline configuration."""
        self.config = {
            "gemini_api_key": "test_key",
            "vqarad_data_path": "test_data",
            "output_dir": "test_output",
            "enable_human_review": True
        }
        self.pipeline = MedVoiceQAPipeline(self.config)
    
    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self):
        """Test complete pipeline execution."""
        # Mock all node operations
        sample_id = "test_001"
        
        # Expected state progression
        initial_state = PipelineState(sample_id=sample_id)
        
        # Mock each node's process method
        with patch.object(self.pipeline.loader, 'process') as mock_loader:
            mock_loader.return_value = {
                "image_path": "test.png",
                "text_query": "Test question?",
                "metadata": {"test": True}
            }
            
            with patch.object(self.pipeline.segmenter, 'process') as mock_segmenter:
                mock_segmenter.return_value = {
                    "visual_box": {"bounding_boxes": []}
                }
                
                with patch.object(self.pipeline.asr_tts, 'process') as mock_asr:
                    mock_asr.return_value = {
                        "speech_path": "test.wav",
                        "asr_text": "Test question",
                        "speech_quality_score": 0.85
                    }
                    
                    with patch.object(self.pipeline.reasoning, 'process') as mock_reasoning:
                        mock_reasoning.return_value = {
                            "text_explanation": "Test explanation",
                            "uncertainty": 0.8
                        }
                        
                        with patch.object(self.pipeline.validation, 'process') as mock_validation:
                            mock_validation.return_value = {
                                "needs_review": False,
                                "critic_notes": "Looks good",
                                "quality_scores": {"overall": 0.9}
                            }
                            
                            # Run pipeline
                            result = await self.pipeline.run_sample(sample_id)
                            
                            # Verify result structure
                            assert "sample_id" in result
                            assert "status" in result
                            assert "final_output" in result
                            assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_pipeline_with_human_review_trigger(self):
        """Test pipeline flow that triggers human review."""
        sample_id = "test_002"
        
        # Mock validation to trigger human review
        with patch.object(self.pipeline.validation, 'process') as mock_validation:
            mock_validation.return_value = {
                "needs_review": True,
                "critic_notes": "Quality concerns identified",
                "quality_scores": {"overall": 0.4}
            }
            
            with patch.object(self.pipeline.human_review, 'process') as mock_review:
                mock_review.return_value = {
                    "review_status": "pending",
                    "review_notes": "Submitted for human review",
                    "reviewer": "pending_assignment"
                }
                
                # Mock other nodes with minimal setup
                with patch.object(self.pipeline.loader, 'process', return_value={"image_path": "test.png", "text_query": "Test", "metadata": {}}):
                    with patch.object(self.pipeline.segmenter, 'process', return_value={"visual_box": {}}):
                        with patch.object(self.pipeline.asr_tts, 'process', return_value={"speech_path": "test.wav", "asr_text": "Test", "speech_quality_score": 0.5}):
                            with patch.object(self.pipeline.reasoning, 'process', return_value={"text_explanation": "Test", "uncertainty": 0.7}):
                                
                                result = await self.pipeline.run_sample(sample_id)
                                
                                # Should trigger human review
                                assert result["status"] in ["needs_review", "pending_review"]
                                assert "human_review" in result


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
