"""
Data sampling utilities for the MedVoiceQAReasonDataset pipeline.
Provides tools for sampling VQA-RAD data and preparing test datasets.
"""

import json
import random
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class VQARADSampler:
    """
    Utility for sampling and preparing VQA-RAD dataset subsets.
    """
    
    def __init__(self, vqarad_path: str, output_dir: str):
        """
        Initialize the sampler.
        
        Args:
            vqarad_path: Path to the VQA-RAD dataset
            output_dir: Output directory for sampled data
        """
        self.vqarad_path = Path(vqarad_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected VQA-RAD structure
        self.images_dir = self.vqarad_path / "images"
        self.questions_file = self.vqarad_path / "questions.json"
        
        if not self.vqarad_path.exists():
            logger.warning(f"VQA-RAD path not found: {self.vqarad_path}")
        
    def sample_balanced_dataset(
        self, 
        n_samples: int = 100, 
        balance_by_modality: bool = True,
        balance_by_answer_type: bool = True,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Create a balanced sample of the VQA-RAD dataset.
        
        Args:
            n_samples: Number of samples to select
            balance_by_modality: Whether to balance by imaging modality (CT, MR, X-ray)
            balance_by_answer_type: Whether to balance by answer type (yes/no, choice, etc.)
            random_seed: Random seed for reproducibility
            
        Returns:
            Dict containing sample metadata and statistics
        """
        random.seed(random_seed)
        
        try:
            # Load questions/answers data
            if self.questions_file.exists():
                with open(self.questions_file, 'r') as f:
                    qa_data = json.load(f)
            else:
                # Create mock data for testing
                qa_data = self._create_mock_qa_data()
            
            # Analyze dataset structure
            analysis = self._analyze_dataset(qa_data)
            logger.info(f"Dataset analysis: {analysis}")
            
            # Perform balanced sampling
            if balance_by_modality or balance_by_answer_type:
                samples = self._balanced_sample(qa_data, n_samples, balance_by_modality, balance_by_answer_type)
            else:
                samples = random.sample(qa_data, min(n_samples, len(qa_data)))
            
            # Copy selected samples
            sample_info = self._prepare_sample_data(samples)
            
            # Save sample metadata
            metadata = {
                "sample_count": len(samples),
                "sampling_method": "balanced" if (balance_by_modality or balance_by_answer_type) else "random",
                "random_seed": random_seed,
                "dataset_analysis": analysis,
                "samples": sample_info,
                "created_timestamp": pd.Timestamp.now().isoformat()
            }
            
            metadata_file = self.output_dir / "sample_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created sample dataset with {len(samples)} samples")
            return metadata
            
        except Exception as e:
            logger.error(f"Error creating sample dataset: {e}")
            return {"error": str(e), "samples": []}
    
    def _analyze_dataset(self, qa_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dataset characteristics."""
        analysis = {
            "total_samples": len(qa_data),
            "modality_distribution": {},
            "answer_type_distribution": {},
            "question_length_stats": {},
            "answer_length_stats": {}
        }
        
        modalities = []
        answer_types = []
        question_lengths = []
        answer_lengths = []
        
        for item in qa_data:
            # Extract modality from image filename or metadata
            image_name = item.get("image", "")
            if "CT" in image_name.upper():
                modality = "CT"
            elif "MR" in image_name.upper() or "MRI" in image_name.upper():
                modality = "MR"
            else:
                modality = "X-ray"  # Default assumption
            modalities.append(modality)
            
            # Classify answer type
            answer = str(item.get("answer", "")).lower()
            if answer in ["yes", "no"]:
                answer_type = "yes_no"
            elif len(answer.split()) == 1:
                answer_type = "single_word"
            elif len(answer.split()) <= 5:
                answer_type = "short_phrase"
            else:
                answer_type = "long_answer"
            answer_types.append(answer_type)
            
            # Length statistics
            question_lengths.append(len(item.get("question", "")))
            answer_lengths.append(len(str(item.get("answer", ""))))
        
        # Compile distributions
        analysis["modality_distribution"] = {mod: modalities.count(mod) for mod in set(modalities)}
        analysis["answer_type_distribution"] = {atype: answer_types.count(atype) for atype in set(answer_types)}
        
        # Length statistics
        if question_lengths:
            analysis["question_length_stats"] = {
                "mean": sum(question_lengths) / len(question_lengths),
                "min": min(question_lengths),
                "max": max(question_lengths)
            }
        
        if answer_lengths:
            analysis["answer_length_stats"] = {
                "mean": sum(answer_lengths) / len(answer_lengths),
                "min": min(answer_lengths),
                "max": max(answer_lengths)
            }
        
        return analysis
    
    def _balanced_sample(
        self, 
        qa_data: List[Dict[str, Any]], 
        n_samples: int,
        balance_by_modality: bool,
        balance_by_answer_type: bool
    ) -> List[Dict[str, Any]]:
        """Perform balanced sampling across specified dimensions."""
        
        # Group samples by categories
        groups = {}
        
        for item in qa_data:
            # Create grouping key
            key_parts = []
            
            if balance_by_modality:
                image_name = item.get("image", "")
                if "CT" in image_name.upper():
                    modality = "CT"
                elif "MR" in image_name.upper():
                    modality = "MR"
                else:
                    modality = "X-ray"
                key_parts.append(modality)
            
            if balance_by_answer_type:
                answer = str(item.get("answer", "")).lower()
                if answer in ["yes", "no"]:
                    answer_type = "yes_no"
                elif len(answer.split()) == 1:
                    answer_type = "single_word"
                elif len(answer.split()) <= 5:
                    answer_type = "short_phrase"
                else:
                    answer_type = "long_answer"
                key_parts.append(answer_type)
            
            group_key = "_".join(key_parts) if key_parts else "all"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
        
        # Sample from each group
        samples_per_group = max(1, n_samples // len(groups))
        selected_samples = []
        
        for group_key, group_items in groups.items():
            group_sample_size = min(samples_per_group, len(group_items))
            group_samples = random.sample(group_items, group_sample_size)
            selected_samples.extend(group_samples)
            logger.info(f"Selected {group_sample_size} samples from group '{group_key}'")
        
        # If we need more samples, randomly select from remaining
        if len(selected_samples) < n_samples:
            remaining = [item for item in qa_data if item not in selected_samples]
            additional_needed = n_samples - len(selected_samples)
            if remaining:
                additional_samples = random.sample(remaining, min(additional_needed, len(remaining)))
                selected_samples.extend(additional_samples)
        
        # If we have too many samples, randomly reduce
        if len(selected_samples) > n_samples:
            selected_samples = random.sample(selected_samples, n_samples)
        
        return selected_samples
    
    def _prepare_sample_data(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare and copy sample data to output directory."""
        sample_info = []
        
        # Create subdirectories
        images_output = self.output_dir / "images"
        images_output.mkdir(exist_ok=True)
        
        for i, sample in enumerate(samples):
            sample_id = f"sample_{i:04d}"
            
            # Copy image if it exists
            image_path = sample.get("image", "")
            source_image_path = self.images_dir / image_path if image_path else None
            
            output_image_path = None
            if source_image_path and source_image_path.exists():
                output_image_path = images_output / f"{sample_id}_{Path(image_path).name}"
                try:
                    shutil.copy2(source_image_path, output_image_path)
                except Exception as e:
                    logger.warning(f"Could not copy image {source_image_path}: {e}")
                    output_image_path = None
            
            # Create sample info
            sample_data = {
                "sample_id": sample_id,
                "original_image": image_path,
                "copied_image": str(output_image_path.relative_to(self.output_dir)) if output_image_path else None,
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "question_type": sample.get("question_type", ""),
                "metadata": {
                    "original_index": i,
                    "has_image": output_image_path is not None
                }
            }
            
            sample_info.append(sample_data)
        
        return sample_info
    
    def _create_mock_qa_data(self) -> List[Dict[str, Any]]:
        """Create mock VQA-RAD data for testing purposes."""
        mock_data = [
            {
                "image": "CT_001.jpg",
                "question": "What organ is primarily visible in this CT scan?",
                "answer": "liver",
                "question_type": "organ_identification"
            },
            {
                "image": "XRAY_002.jpg", 
                "question": "Is there evidence of pneumonia in this chest X-ray?",
                "answer": "yes",
                "question_type": "yes_no"
            },
            {
                "image": "MR_003.jpg",
                "question": "What abnormality is shown in this MRI?",
                "answer": "brain tumor in the frontal lobe",
                "question_type": "abnormality_detection"
            },
            {
                "image": "CT_004.jpg",
                "question": "How many vertebrae are visible?",
                "answer": "7",
                "question_type": "counting"
            },
            {
                "image": "XRAY_005.jpg",
                "question": "Is the heart size normal?",
                "answer": "no",
                "question_type": "yes_no"
            }
        ]
        
        # Expand with variations
        expanded_data = []
        for i in range(50):  # Create 50 mock samples
            base_sample = mock_data[i % len(mock_data)]
            sample = base_sample.copy()
            sample["image"] = f"{sample['image'].split('_')[0]}_{i:03d}.jpg"
            expanded_data.append(sample)
        
        logger.info(f"Created {len(expanded_data)} mock samples for testing")
        return expanded_data


class DatasetValidator:
    """
    Validate dataset integrity and pipeline readiness.
    """
    
    @staticmethod
    def validate_sample_dataset(dataset_dir: str) -> Dict[str, Any]:
        """
        Validate a sample dataset for pipeline compatibility.
        
        Args:
            dataset_dir: Path to sample dataset directory
            
        Returns:
            Validation report
        """
        dataset_path = Path(dataset_dir)
        
        validation_report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # Check directory structure
            required_files = ["sample_metadata.json"]
            for req_file in required_files:
                if not (dataset_path / req_file).exists():
                    validation_report["errors"].append(f"Missing required file: {req_file}")
                    validation_report["valid"] = False
            
            # Load and validate metadata
            metadata_file = dataset_path / "sample_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check metadata structure
                required_fields = ["sample_count", "samples"]
                for field in required_fields:
                    if field not in metadata:
                        validation_report["errors"].append(f"Missing metadata field: {field}")
                        validation_report["valid"] = False
                
                # Validate samples
                samples = metadata.get("samples", [])
                validation_report["statistics"]["declared_sample_count"] = metadata.get("sample_count", 0)
                validation_report["statistics"]["actual_sample_count"] = len(samples)
                
                # Check sample structure
                missing_images = 0
                for sample in samples:
                    required_sample_fields = ["sample_id", "question", "answer"]
                    for field in required_sample_fields:
                        if field not in sample:
                            validation_report["warnings"].append(f"Sample {sample.get('sample_id', 'unknown')} missing field: {field}")
                    
                    # Check image availability
                    if sample.get("copied_image"):
                        image_path = dataset_path / sample["copied_image"]
                        if not image_path.exists():
                            missing_images += 1
                
                if missing_images > 0:
                    validation_report["warnings"].append(f"{missing_images} samples have missing image files")
                
                validation_report["statistics"]["missing_images"] = missing_images
                validation_report["statistics"]["samples_with_images"] = len(samples) - missing_images
            
        except Exception as e:
            validation_report["errors"].append(f"Validation error: {str(e)}")
            validation_report["valid"] = False
        
        return validation_report


def create_test_dataset(output_dir: str = "data/test_sample", n_samples: int = 10) -> Dict[str, Any]:
    """
    Create a small test dataset for pipeline development and testing.
    
    Args:
        output_dir: Output directory for test dataset
        n_samples: Number of samples to create
        
    Returns:
        Test dataset metadata
    """
    sampler = VQARADSampler("data/vqarad", output_dir)
    
    # Create balanced test dataset
    result = sampler.sample_balanced_dataset(
        n_samples=n_samples,
        balance_by_modality=True,
        balance_by_answer_type=True,
        random_seed=42
    )
    
    # Validate the created dataset
    validation = DatasetValidator.validate_sample_dataset(output_dir)
    result["validation"] = validation
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample VQA-RAD dataset for pipeline testing")
    parser.add_argument("--input", "-i", default="data/vqarad", help="Input VQA-RAD dataset path")
    parser.add_argument("--output", "-o", default="data/sample", help="Output directory")
    parser.add_argument("--samples", "-n", type=int, default=50, help="Number of samples")
    parser.add_argument("--balance-modality", action="store_true", help="Balance by imaging modality")
    parser.add_argument("--balance-answer", action="store_true", help="Balance by answer type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sampler
    sampler = VQARADSampler(args.input, args.output)
    
    # Create sample dataset
    result = sampler.sample_balanced_dataset(
        n_samples=args.samples,
        balance_by_modality=args.balance_modality,
        balance_by_answer_type=args.balance_answer,
        random_seed=args.seed
    )
    
    print(f"Created sample dataset in {args.output}")
    print(f"Samples: {result.get('sample_count', 0)}")
    print(f"Method: {result.get('sampling_method', 'unknown')}")
    
    # Validate dataset
    validation = DatasetValidator.validate_sample_dataset(args.output)
    if validation["valid"]:
        print("✓ Dataset validation passed")
    else:
        print("✗ Dataset validation failed:")
        for error in validation["errors"]:
            print(f"  - {error}")
        for warning in validation["warnings"]:
            print(f"  ! {warning}")
