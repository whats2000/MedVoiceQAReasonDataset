"""
Pytest configuration and fixtures for MedVoiceQA tests.
"""

import asyncio
import logging
import pytest
import sys
from pathlib import Path

# Add project root to Python path for all tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def hf_data_dir():
    """Fixture providing the Hugging Face data directory path."""
    return Path("data/vqarad_hf")


@pytest.fixture(scope="session")
def project_root_path():
    """Fixture providing the project root path."""
    return project_root


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        force=True
    )


@pytest.fixture
def sample_pipeline_state():
    """Fixture providing a sample pipeline state for testing."""
    return {
        "sample_id": "test_sample_001",
        "text_query": "What abnormality is shown in this chest X-ray?",
        "image_path": None,
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


@pytest.fixture(scope="session")
def hf_loader():
    """Fixture providing a HuggingFace loader instance."""
    from data.huggingface_loader import HuggingFaceVQARADLoader
    return HuggingFaceVQARADLoader()
