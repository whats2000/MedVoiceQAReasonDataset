"""
MedVoiceQA Reasoning Dataset Pipeline

A LangGraph-based pipeline for transforming VQA-RAD into a multi-modal,
explainable medical QA dataset with speech, visual localization, and reasoning.
"""

from .graph import create_medvoice_pipeline, get_pipeline_info
from .run_pipeline import run_pipeline

__version__ = "0.1.0"
__author__ = "MedVoiceQA Team"

__all__ = ["create_medvoice_pipeline", "get_pipeline_info", "run_pipeline"]
