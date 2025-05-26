#!/usr/bin/env python3
"""
Test Hugging Face VQA-RAD integration without requiring API keys.

This script validates that we can successfully load samples from the 
Hugging Face VQA-RAD dataset that we've already downloaded.
"""

import asyncio
import json
import logging
import pytest
from pathlib import Path
from data.huggingface_loader import HuggingFaceVQARADLoader

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_hf_data_directory_exists(hf_data_dir):
    """Test that Hugging Face data directory exists."""
    assert hf_data_dir.exists(), "Hugging Face data directory not found!"
    logger.info("✅ Hugging Face data directory found")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_hf_loader_initialization(hf_loader):
    """Test HuggingFaceVQARADLoader initialization."""
    assert hf_loader is not None
    logger.info("✅ HuggingFaceVQARADLoader initialized successfully")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_hf_data_loading(hf_data_dir, hf_loader):
    """Test loading samples from the existing Hugging Face data."""
    logger.info("Testing Hugging Face VQA-RAD data loading...")

    # Check if HF data exists
    if not hf_data_dir.exists():
        pytest.skip("Hugging Face data directory not found!")

    # Test loading samples concurrently for speed
    samples_to_test = 5
    
    async def load_sample(index):
        try:
            return hf_loader.get_sample_by_index(index)
        except Exception as e:
            logger.error(f"Error loading sample {index}: {e}")
            return None

    # Load samples concurrently
    tasks = [load_sample(i) for i in range(samples_to_test)]
    loaded_samples = await asyncio.gather(*tasks)
    
    # Filter out None results
    loaded_samples = [sample for sample in loaded_samples if sample is not None]

    assert len(loaded_samples) > 0, "No samples could be loaded"

    for i, sample in enumerate(loaded_samples):
        logger.info(f"Sample {i}: {sample['sample_id']} - {sample['question'][:50]}...")
        logger.info(f"  Answer: {sample['answer']}")
        logger.info(f"  Image: {sample['image_path']}")
        logger.info(f"  Metadata: {list(sample.get('metadata', {}).keys())}")

    logger.info(f"✅ Successfully loaded {len(loaded_samples)} samples from Hugging Face data")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sample_structure(hf_loader):
    """Test sample structure validation."""
    sample = hf_loader.get_sample_by_index(0)
    
    if sample is None:
        pytest.skip("No sample available for testing")

    required_fields = ['sample_id', 'question', 'answer', 'image_path']
    missing_fields = [field for field in required_fields if field not in sample]

    assert not missing_fields, f"Missing required fields: {missing_fields}"
    logger.info("✅ Sample structure validation passed")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_image_files_exist(hf_loader):
    """Test that image files exist for loaded samples."""
    samples_to_check = 3
    
    for i in range(samples_to_check):
        sample = hf_loader.get_sample_by_index(i)
        if sample is None:
            continue
            
        image_path = Path(sample['image_path'])
        assert image_path.exists(), f"Image file not found: {image_path}"

    logger.info("✅ All image files found")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_metadata_structure():
    """Test the metadata file structure."""
    metadata_file = Path("data/vqarad_hf/sample_metadata.json")
    
    if not metadata_file.exists():
        pytest.skip("Metadata file not found")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    required_keys = ['sample_count', 'samples']
    missing_keys = [key for key in required_keys if key not in metadata]

    assert not missing_keys, f"Missing metadata keys: {missing_keys}"

    logger.info(f"✅ Metadata structure valid - {metadata['sample_count']} samples available")

    # Check if the data has the additional analysis information
    if 'modalities' in metadata:
        logger.info(f"  Modalities: {metadata['modalities']}")
    if 'question_types' in metadata:
        logger.info(f"  Question types: {metadata['question_types']}")

    # Look at first few samples to understand structure
    if metadata['samples']:
        sample = metadata['samples'][0]
        logger.info(f"  Sample fields: {list(sample.keys())}")
        if 'metadata' in sample:
            logger.info(f"  Sample metadata fields: {list(sample['metadata'].keys())}")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_complete_hf_integration(hf_data_dir, hf_loader):
    """Test complete Hugging Face integration - combines all tests."""
    logger.info("🧪 Testing Hugging Face VQA-RAD Integration")
    logger.info("=" * 50)

    # Run all tests concurrently for speed
    tasks = [
        test_hf_data_directory_exists(hf_data_dir),
        test_hf_loader_initialization(hf_loader),
        test_metadata_structure(),
        test_sample_structure(hf_loader),
        test_image_files_exist(hf_loader),
        test_hf_data_loading(hf_data_dir, hf_loader)
    ]
    
    await asyncio.gather(*tasks)
    
    logger.info("🎉 All Hugging Face integration tests passed!")
    logger.info("Ready to run the full pipeline with real VQA-RAD data")
