"""
Prompt management system for MedVoiceQA pipeline nodes.

This module provides functionality to load and validate prompts from text files,
with fallback to default prompts when custom prompts are invalid or missing.
"""

from .prompt_manager import PromptManager

__all__ = ["PromptManager"]
