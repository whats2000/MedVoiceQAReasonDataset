"""
Test cases for the prompt management system.
"""

import pytest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prompts import PromptManager


class TestPromptManager:
    """Test cases for the PromptManager class."""
    
    def test_init(self):
        """Test PromptManager initialization."""
        pm = PromptManager()
        assert pm.prompts_dir.exists()
        assert pm._prompt_cache == {}
    
    def test_get_prompt_with_default(self):
        """Test getting a prompt that falls back to default."""
        pm = PromptManager()
        
        default_prompt = "Default prompt with {var1}"
        result = pm.get_prompt(
            "non_existent_prompt",
            default_prompt,
            {"var1"}
        )
        
        assert result == default_prompt
    
    def test_get_prompt_validation_success(self):
        """Test prompt validation with all required variables present."""
        with TemporaryDirectory() as temp_dir:
            pm = PromptManager(temp_dir)
            
            # Create a custom prompt file
            prompt_file = Path(temp_dir) / "test_prompt.txt"
            custom_prompt = "Custom prompt with {var1} and {var2}"
            prompt_file.write_text(custom_prompt, encoding='utf-8')
            
            result = pm.get_prompt(
                "test_prompt",
                "Default prompt",
                {"var1", "var2"}
            )
            
            assert result == custom_prompt
    
    def test_get_prompt_validation_failure(self):
        """Test prompt validation failure when required variables are missing."""
        with TemporaryDirectory() as temp_dir:
            pm = PromptManager(temp_dir)
            
            # Create a custom prompt file missing required variables
            prompt_file = Path(temp_dir) / "test_prompt.txt"
            custom_prompt = "Custom prompt with {var1}"  # missing var2
            prompt_file.write_text(custom_prompt, encoding='utf-8')
            
            default_prompt = "Default prompt with {var1} and {var2}"
            result = pm.get_prompt(
                "test_prompt",
                default_prompt,
                {"var1", "var2"}  # var2 is missing from custom prompt
            )
            
            # Should fall back to default
            assert result == default_prompt
    
    def test_get_prompt_no_validation_required(self):
        """Test getting a prompt when no validation is required."""
        with TemporaryDirectory() as temp_dir:
            pm = PromptManager(temp_dir)
            
            prompt_file = Path(temp_dir) / "test_prompt.txt"
            custom_prompt = "Custom prompt without variables"
            prompt_file.write_text(custom_prompt, encoding='utf-8')
            
            result = pm.get_prompt(
                "test_prompt",
                "Default prompt",
                None  # No validation required
            )
            
            assert result == custom_prompt
    
    def test_load_custom_prompt_file_not_found(self):
        """Test loading a custom prompt when file doesn't exist."""
        pm = PromptManager()
        
        result = pm._load_custom_prompt("definitely_does_not_exist")
        assert result is None
    
    def test_load_custom_prompt_empty_file(self):
        """Test loading a custom prompt from an empty file."""
        with TemporaryDirectory() as temp_dir:
            pm = PromptManager(temp_dir)
            
            # Create empty prompt file
            prompt_file = Path(temp_dir) / "empty_prompt.txt"
            prompt_file.write_text("", encoding='utf-8')
            
            result = pm._load_custom_prompt("empty_prompt")
            assert result is None
    
    def test_validate_prompt_success(self):
        """Test prompt validation with all required variables."""
        pm = PromptManager()
        
        prompt = "This is a prompt with {var1} and {var2}"
        required_vars = {"var1", "var2"}
        
        assert pm._validate_prompt(prompt, required_vars, "test") is True
    
    def test_validate_prompt_failure(self):
        """Test prompt validation with missing variables."""
        pm = PromptManager()
        
        prompt = "This is a prompt with {var1}"
        required_vars = {"var1", "var2"}  # var2 is missing
        
        assert pm._validate_prompt(prompt, required_vars, "test") is False
    
    def test_validate_prompt_no_requirements(self):
        """Test prompt validation when no variables are required."""
        pm = PromptManager()
        
        prompt = "This is a prompt without variables"
        required_vars = set()
        
        assert pm._validate_prompt(prompt, required_vars, "test") is True
    
    def test_create_template_prompt(self):
        """Test creating a template prompt file."""
        with TemporaryDirectory() as temp_dir:
            pm = PromptManager(temp_dir)
            
            default_prompt = "Template prompt with {var1}"
            pm.create_template_prompt("template_test", default_prompt)
            
            # Check if file was created
            prompt_file = Path(temp_dir) / "template_test.txt"
            assert prompt_file.exists()
            assert prompt_file.read_text(encoding='utf-8') == default_prompt
    
    def test_create_template_prompt_file_exists(self):
        """Test creating a template when file already exists."""
        with TemporaryDirectory() as temp_dir:
            pm = PromptManager(temp_dir)
            
            # Create file first
            prompt_file = Path(temp_dir) / "existing_prompt.txt"
            original_content = "Original content"
            prompt_file.write_text(original_content, encoding='utf-8')
            
            # Try to create template (should not overwrite)
            pm.create_template_prompt("existing_prompt", "New content")
            
            # Original content should remain
            assert prompt_file.read_text(encoding='utf-8') == original_content
    
    def test_list_available_prompts(self):
        """Test listing available prompt files."""
        with TemporaryDirectory() as temp_dir:
            pm = PromptManager(temp_dir)
            
            # Create some prompt files
            (Path(temp_dir) / "prompt1.txt").write_text("Content 1", encoding='utf-8')
            (Path(temp_dir) / "prompt2.txt").write_text("Content 2", encoding='utf-8')
            (Path(temp_dir) / "not_a_prompt.md").write_text("Not a prompt", encoding='utf-8')
            
            available = pm.list_available_prompts()
            
            assert "prompt1" in available
            assert "prompt2" in available
            assert "not_a_prompt" not in available  # Should only include .txt files
            assert available["prompt1"] is True
            assert available["prompt2"] is True
    
    def test_clear_cache(self):
        """Test clearing the prompt cache."""
        pm = PromptManager()
        
        # Add something to cache
        pm._prompt_cache["test_key"] = "test_value"
        assert len(pm._prompt_cache) > 0
        
        # Clear cache
        pm.clear_cache()
        assert len(pm._prompt_cache) == 0


def test_integration_with_actual_prompts():
    """Integration test with the actual prompt files in the project."""
    pm = PromptManager()
    
    # Test loading actual prompt files that should exist
    segmentation_prompt = pm.get_prompt(
        "segmentation",
        "fallback",
        {"text_query"}
    )
    
    explanation_prompt = pm.get_prompt(
        "explanation", 
        "fallback",
        {"query", "visual_box"}
    )
    
    validation_prompt = pm.get_prompt(
        "validation",
        "fallback", 
        {"query", "visual_box", "speech_quality", "asr_text", "explanation", "uncertainty"}
    )
    
    # These should not be the fallback since the files exist
    assert segmentation_prompt != "fallback"
    assert explanation_prompt != "fallback"
    assert validation_prompt != "fallback"
    
    # Check that required variables are present
    assert "{text_query}" in segmentation_prompt
    assert "{query}" in explanation_prompt
    assert "{visual_box}" in explanation_prompt
    assert all(var in validation_prompt for var in ["{query}", "{visual_box}", "{speech_quality}", "{asr_text}", "{explanation}", "{uncertainty}"])


if __name__ == "__main__":
    # Run a simple test if executed directly
    test_integration_with_actual_prompts()
    print("✅ Integration test passed!")
