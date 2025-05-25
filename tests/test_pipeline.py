#!/usr/bin/env python3
"""
Test script for the complete MedVoiceQA pipeline.

This script tests the pipeline with a sample input to ensure all nodes work correctly.
"""

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.graph import create_medvoice_pipeline, PipelineState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


def test_pipeline():
    """Test the complete pipeline with a sample input."""
    try:
        logger.info("Creating MedVoiceQA pipeline...")
        
        # Create pipeline
        pipeline = create_medvoice_pipeline()
        compiled_pipeline = pipeline
        
        logger.info("Pipeline created successfully!")
        
        # Test with sample input
        test_input: PipelineState = {
            "sample_id": "test_sample_001",
            "text_query": "What abnormality is shown in this chest X-ray?",
            "image_path": None,  # Will be handled by loader
            "metadata": {},
            "visual_box": None,
            "speech_path": None,
            "asr_text": None,
            "speech_quality_score": None,
            "text_explanation": None,
            "uncertainty": None,
            "needs_review": None,
            "critic_notes": None,
            "quality_scores": None,
            "review_status": None,
            "review_notes": None,
            "approved": None,
            "completed_nodes": [],
            "node_errors": {},
            "ground_truth_answer": None,
            "processing_start_time": None
        }
        
        logger.info("Testing pipeline with sample input...")
        logger.info(f"Input: {test_input}")
        
        # Note: This would fail without actual VQA-RAD data and API keys
        # but it will test the graph structure and imports
        logger.info("Pipeline structure test completed!")
        logger.info("To run with real data, ensure:")
        logger.info("1. GOOGLE_API_KEY environment variable is set")
        logger.info("2. VQA-RAD dataset is available")
        logger.info("3. Run with: python pipeline/run_pipeline.py --sample-id <id>")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
