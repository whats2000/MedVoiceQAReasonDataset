"""
Validation node package for MedVoiceQA pipeline.

Provides quality assessment and critic validation using Gemini models.
"""

from .validation import GeminiValidationDuo, run_validation

__all__ = [
    "GeminiValidationDuo",
    "run_validation"
]
