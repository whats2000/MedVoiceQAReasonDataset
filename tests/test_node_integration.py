"""
Integration tests for the prompt management system with pipeline nodes.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prompts import PromptManager


class TestNodePromptIntegration:
    """Test integration between nodes and the prompt management system."""
    
    def test_segmentation_node_import(self):
        """Test that segmentation node can import and use PromptManager."""
        try:
            from nodes.segmentation.segmentation import GeminiVisionSegmenter
            
            # Mock the Gemini client to avoid API calls
            with patch('nodes.segmentation.segmentation.genai.Client'):
                with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
                    segmenter = GeminiVisionSegmenter()
                    
                    # Verify prompt manager is initialized
                    assert hasattr(segmenter, 'prompt_manager')
                    assert isinstance(segmenter.prompt_manager, PromptManager)
                    
                    # Test prompt creation
                    prompt = segmenter._create_segmentation_prompt("test query")
                    assert isinstance(prompt, str)
                    assert "test query" in prompt
                    
        except ImportError as e:
            pytest.skip(f"Segmentation node not available: {e}")
    
    def test_explanation_node_import(self):
        """Test that explanation node can import and use PromptManager."""
        try:
            from nodes.explanation.explanation import GeminiReasoningEngine
            
            # Mock the Gemini client to avoid API calls
            with patch('nodes.explanation.explanation.genai.Client'):
                with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
                    engine = GeminiReasoningEngine()
                    
                    # Verify prompt manager is initialized
                    assert hasattr(engine, 'prompt_manager')
                    assert isinstance(engine.prompt_manager, PromptManager)
                    
        except ImportError as e:
            pytest.skip(f"Explanation node not available: {e}")
    
    def test_validation_node_import(self):
        """Test that validation node can import and use PromptManager."""
        try:
            from nodes.validation.validation import GeminiValidationDuo
            
            # Mock the Gemini client to avoid API calls
            with patch('nodes.validation.validation.genai.Client'):
                with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
                    validator = GeminiValidationDuo()
                    
                    # Verify prompt manager is initialized
                    assert hasattr(validator, 'prompt_manager')
                    assert isinstance(validator.prompt_manager, PromptManager)
                    
        except ImportError as e:
            pytest.skip(f"Validation node not available: {e}")
    
    def test_all_required_prompt_files_exist(self):
        """Test that all required prompt files exist and are valid."""
        pm = PromptManager()
        
        # Test segmentation prompt
        segmentation_prompt = pm.get_prompt(
            "segmentation",
            "fallback",
            {"text_query"}
        )
        assert segmentation_prompt != "fallback"
        assert "{text_query}" in segmentation_prompt
        
        # Test explanation prompt
        explanation_prompt = pm.get_prompt(
            "explanation",
            "fallback", 
            {"query", "visual_box"}
        )
        assert explanation_prompt != "fallback"
        assert "{query}" in explanation_prompt
        assert "{visual_box}" in explanation_prompt
        
        # Test validation prompt
        validation_prompt = pm.get_prompt(
            "validation",
            "fallback",
            {"query", "visual_box", "speech_quality", "asr_text", "explanation", "uncertainty"}
        )
        assert validation_prompt != "fallback"
        for var in ["query", "visual_box", "speech_quality", "asr_text", "explanation", "uncertainty"]:
            assert f"{{{var}}}" in validation_prompt
    
    def test_custom_prompt_override(self):
        """Test that custom prompts can override default ones."""
        from tempfile import TemporaryDirectory
        
        with TemporaryDirectory() as temp_dir:
            pm = PromptManager(temp_dir)
            
            # Create a custom segmentation prompt
            custom_prompt = "Custom segmentation prompt with {text_query} variable."
            prompt_file = Path(temp_dir) / "segmentation.txt"
            prompt_file.write_text(custom_prompt, encoding='utf-8')
            
            # Test that custom prompt is loaded
            result = pm.get_prompt(
                "segmentation",
                "Default prompt",
                {"text_query"}
            )
            
            assert result == custom_prompt
    
    def test_prompt_validation_prevents_broken_prompts(self):
        """Test that prompt validation prevents broken prompts from being used."""
        from tempfile import TemporaryDirectory
        
        with TemporaryDirectory() as temp_dir:
            pm = PromptManager(temp_dir)
            
            # Create a broken custom prompt (missing required variable)
            broken_prompt = "Broken prompt without the required variable."
            prompt_file = Path(temp_dir) / "segmentation.txt"
            prompt_file.write_text(broken_prompt, encoding='utf-8')
            
            # Test that default prompt is used instead
            default_prompt = "Default prompt with {text_query}"
            result = pm.get_prompt(
                "segmentation",
                default_prompt,
                {"text_query"}
            )
            
            assert result == default_prompt  # Should fallback to default
            assert result != broken_prompt   # Should not use broken prompt


def test_prompt_files_content_quality():
    """Test the quality and structure of the actual prompt files."""
    pm = PromptManager()
    
    # Test segmentation prompt structure
    seg_prompt = pm._load_custom_prompt("segmentation")
    if seg_prompt:
        assert "medical" in seg_prompt.lower()
        assert "bounding box" in seg_prompt.lower()
        assert "{text_query}" in seg_prompt
    
    # Test explanation prompt structure
    exp_prompt = pm._load_custom_prompt("explanation")
    if exp_prompt:
        assert "medical" in exp_prompt.lower()
        assert "reasoning" in exp_prompt.lower()
        assert "{query}" in exp_prompt
        assert "{visual_box}" in exp_prompt
    
    # Test validation prompt structure
    val_prompt = pm._load_custom_prompt("validation")
    if val_prompt:
        assert "quality" in val_prompt.lower()
        assert "assessment" in val_prompt.lower()
        assert "{query}" in val_prompt


if __name__ == "__main__":
    # Run basic integration test if executed directly
    test_prompt_files_content_quality()
    print("✅ Prompt content quality test passed!")
