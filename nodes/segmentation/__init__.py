"""
Segmentation node package for visual localization using Gemini Vision.
"""

from .segmentation import run_segmentation, GeminiVisionSegmenter

__all__ = ["run_segmentation", "GeminiVisionSegmenter"]
