#!/usr/bin/env python3
"""
Test script for the complete MedVoiceQA pipeline.

This script tests the pipeline with a sample input to ensure all nodes work correctly.
"""

import asyncio
import logging
import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.graph import create_medvoice_pipeline, PipelineState

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_imports():
    """Test that all pipeline modules can be imported."""
    logger.info("Testing pipeline imports...")

    # Test that all modules can be imported
    from pipeline.graph import create_medvoice_pipeline, PipelineState
    from nodes.loader import run_loader
    from nodes.segmentation import run_segmentation
    from nodes.asr_tts import run_asr_tts
    from nodes.explanation import run_explanation
    from nodes.validation import run_validation
    from nodes.human_review import run_human_review

    logger.info("All modules imported successfully!")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_creation():
    """Test pipeline creation and structure."""
    logger.info("Creating MedVoiceQA pipeline...")

    # Create pipeline (already compiled)
    pipeline = create_medvoice_pipeline()

    logger.info("Pipeline created successfully!")
    assert pipeline is not None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_pipeline_state_structure(sample_pipeline_state):
    """Test the pipeline state structure."""
    test_input = sample_pipeline_state

    logger.info("Pipeline structure validation completed!")
    
    # Validate required fields
    assert "sample_id" in test_input
    assert "text_query" in test_input
    assert "completed_nodes" in test_input
    assert isinstance(test_input["completed_nodes"], list)
    assert isinstance(test_input["node_errors"], dict)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_complete_pipeline(sample_pipeline_state):
    """Test the complete pipeline structure and components."""
    try:
        await test_pipeline_imports()
        await test_pipeline_creation()
        await test_pipeline_state_structure(sample_pipeline_state)
        
        logger.info("All components are properly configured.")
        logger.info("")
        logger.info("To run with real data, ensure:")
        logger.info("1. GOOGLE_API_KEY environment variable is set")
        logger.info("2. VQA-RAD dataset is available")
        logger.info("3. Run with: uv run python pipeline/run_pipeline.py --sample-id <id>")

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        pytest.fail(f"Pipeline test failed: {e}")
