#!/usr/bin/env python3
"""
Test Hugging Face VQA-RAD integration without requiring API keys.

This script validates that we can successfully load samples from the 
Hugging Face VQA-RAD dataset that we've already downloaded.
"""

import logging
from pathlib import Path
from data.huggingface_loader import HuggingFaceVQARADLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_hf_data_loading():
    """Test loading samples from the existing Hugging Face data."""
    try:
        logger.info("Testing Hugging Face VQA-RAD data loading...")
        
        # Check if HF data exists
        hf_data_dir = Path("data/vqarad_hf")
        if not hf_data_dir.exists():
            logger.error("Hugging Face data directory not found!")
            return False
        
        # Initialize the loader
        hf_loader = HuggingFaceVQARADLoader()
        logger.info("HuggingFaceVQARADLoader initialized successfully")
        
        # Test loading a few samples
        samples_to_test = 5
        loaded_samples = []
        
        for i in range(samples_to_test):
            try:
                sample = hf_loader.get_sample(i)
                if sample:
                    loaded_samples.append(sample)
                    logger.info(f"Sample {i}: {sample['sample_id']} - {sample['question'][:50]}...")
                    logger.info(f"  Answer: {sample['answer']}")
                    logger.info(f"  Image: {sample['image_path']}")
                    logger.info(f"  Metadata: {list(sample.get('metadata', {}).keys())}")
                else:
                    logger.warning(f"No sample found at index {i}")
                    break
            except Exception as e:
                logger.error(f"Error loading sample {i}: {e}")
                continue
        
        if loaded_samples:
            logger.info(f"✅ Successfully loaded {len(loaded_samples)} samples from Hugging Face data")
            
            # Test sample structure
            sample = loaded_samples[0]
            required_fields = ['sample_id', 'question', 'answer', 'image_path']
            missing_fields = [field for field in required_fields if field not in sample]
            
            if missing_fields:
                logger.error(f"Missing required fields: {missing_fields}")
                return False
            
            logger.info("✅ Sample structure validation passed")
            
            # Check if image files exist
            for sample in loaded_samples:
                image_path = Path(sample['image_path'])
                if not image_path.exists():
                    logger.error(f"Image file not found: {image_path}")
                    return False
            
            logger.info("✅ All image files found")
            return True
        else:
            logger.error("No samples could be loaded")
            return False
            
    except Exception as e:
        logger.error(f"HF data loading test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_metadata_structure():
    """Test the metadata file structure."""
    try:
        metadata_file = Path("data/vqarad_hf/sample_metadata.json")
        if not metadata_file.exists():
            logger.error("Metadata file not found")
            return False
        
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        required_keys = ['total_samples', 'samples', 'modalities', 'question_types']
        missing_keys = [key for key in required_keys if key not in metadata]
        
        if missing_keys:
            logger.error(f"Missing metadata keys: {missing_keys}")
            return False
        
        logger.info(f"✅ Metadata structure valid - {metadata['total_samples']} samples available")
        logger.info(f"  Modalities: {metadata['modalities']}")
        logger.info(f"  Question types: {metadata['question_types']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Metadata test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("🧪 Testing Hugging Face VQA-RAD Integration")
    logger.info("=" * 50)
    
    success = True
    
    # Test metadata structure
    if not test_metadata_structure():
        success = False
    
    # Test data loading
    if not test_hf_data_loading():
        success = False
    
    if success:
        logger.info("🎉 All Hugging Face integration tests passed!")
        logger.info("Ready to run the full pipeline with real VQA-RAD data")
    else:
        logger.error("❌ Some tests failed")
    
    exit(0 if success else 1)
