"""
Prompt Manager for MedVoiceQA Pipeline

Handles loading, validation, and management of prompts for different pipeline nodes.
Supports custom prompts with fallback to default prompts when validation fails.
"""

import logging
import re
import warnings
from pathlib import Path
from typing import Dict, Any, Set, Optional, Union

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompts for pipeline nodes with validation and fallback capabilities.
    
    Features:
    - Load custom prompts from text files
    - Validate prompts contain required variables
    - Fallback to default prompts when validation fails
    - Warning system for invalid prompts
    """
    
    def __init__(self, prompts_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt files. Defaults to './prompts'
        """
        if prompts_dir is None:
            # Get the project root (assuming this file is in prompts/ subdirectory)
            self.prompts_dir = Path(__file__).parent
        else:
            self.prompts_dir = Path(prompts_dir)
            
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Cache for loaded prompts
        self._prompt_cache: Dict[str, str] = {}
        
        logger.info(f"PromptManager initialized with directory: {self.prompts_dir}")
    
    def get_prompt(
        self, 
        prompt_name: str, 
        default_prompt: str, 
        required_variables: Optional[Set[str]] = None
    ) -> str:
        """
        Get a prompt by name with validation and fallback.
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            default_prompt: Default prompt to use if custom prompt is invalid
            required_variables: Set of variable names that must be present in the prompt
            
        Returns:
            The validated prompt text
        """
        if required_variables is None:
            required_variables = set()
            
        # Check cache first
        cache_key = f"{prompt_name}_{hash(frozenset(required_variables))}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        
        # Try to load custom prompt
        custom_prompt = self._load_custom_prompt(prompt_name)
        
        if custom_prompt is not None:
            # Validate custom prompt
            if self._validate_prompt(custom_prompt, required_variables, prompt_name):
                self._prompt_cache[cache_key] = custom_prompt
                logger.info(f"Using custom prompt: {prompt_name}")
                return custom_prompt
            else:
                logger.warning(f"Custom prompt '{prompt_name}' failed validation, using default")
        
        # Validate default prompt as well (for safety)
        if not self._validate_prompt(default_prompt, required_variables, f"{prompt_name}_default"):
            logger.error(f"Default prompt for '{prompt_name}' is invalid! This is a code bug.")
            # Still return it, but log the error
        
        self._prompt_cache[cache_key] = default_prompt
        return default_prompt
    
    def _load_custom_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Load a custom prompt from file.
        
        Args:
            prompt_name: Name of the prompt file (without extension)
            
        Returns:
            Prompt content or None if file doesn't exist or can't be read
        """
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            logger.debug(f"Custom prompt file not found: {prompt_file}")
            return None
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                logger.warning(f"Custom prompt file is empty: {prompt_file}")
                return None
                
            logger.debug(f"Loaded custom prompt from: {prompt_file}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to read prompt file {prompt_file}: {e}")
            return None
    
    def _validate_prompt(
        self, 
        prompt: str, 
        required_variables: Set[str], 
        prompt_name: str
    ) -> bool:
        """
        Validate that a prompt contains all required variables.
        
        Args:
            prompt: The prompt text to validate
            required_variables: Set of variable names that must be present
            prompt_name: Name of the prompt (for logging)
            
        Returns:
            True if valid, False otherwise
        """
        if not required_variables:
            return True  # No validation needed
        
        # Find all variables in the prompt using regex
        # Look for {variable_name} patterns
        found_variables = set(re.findall(r'\{([^}]+)\}', prompt))
        
        missing_variables = required_variables - found_variables
        
        if missing_variables:
            logger.warning(
                f"Prompt '{prompt_name}' is missing required variables: {missing_variables}"
            )
            return False
        
        logger.debug(f"Prompt '{prompt_name}' validation passed")
        return True
    
    def create_template_prompt(self, prompt_name: str, default_prompt: str) -> None:
        """
        Create a template prompt file from the default prompt.
        
        Args:
            prompt_name: Name for the prompt file
            default_prompt: Default prompt content to use as template
        """
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        if prompt_file.exists():
            logger.info(f"Prompt file already exists: {prompt_file}")
            return
        
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(default_prompt)
            
            logger.info(f"Created template prompt file: {prompt_file}")
            
        except Exception as e:
            logger.error(f"Failed to create template prompt file {prompt_file}: {e}")
    
    def list_available_prompts(self) -> Dict[str, bool]:
        """
        List all available prompt files.
        
        Returns:
            Dict mapping prompt names to whether they exist as custom files
        """
        prompt_files = {}
        
        # Find all .txt files in prompts directory
        for prompt_file in self.prompts_dir.glob("*.txt"):
            prompt_name = prompt_file.stem
            prompt_files[prompt_name] = True
        
        return prompt_files
    
    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._prompt_cache.clear()
        logger.debug("Prompt cache cleared")


# Global instance for easy access
prompt_manager = PromptManager()
