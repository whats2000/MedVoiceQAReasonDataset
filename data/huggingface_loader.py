"""
Hugging Face VQA-RAD Dataset Loader

Loads the VQA-RAD dataset from Hugging Face and provides utilities
for sampling and preprocessing for the MedVoiceQA pipeline.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pandas as pd
from PIL import Image
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


class HuggingFaceVQARADLoader:
    """
    Loader for VQA-RAD dataset from Hugging Face.
    """

    def __init__(self, cache_dir: Optional[str] = None, output_dir: str = "data/vqarad_hf"):
        """
        Initialize the Hugging Face VQA-RAD loader.
        
        Args:
            cache_dir: Optional cache directory for Hugging Face datasets
            output_dir: Output directory for processed data
        """
        self.cache_dir = cache_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.dataset: Optional[Union[Dataset, Any]] = None
        self._loaded = False

    def load_dataset(self, split: str = "train") -> None:
        """
        Load the VQA-RAD dataset from Hugging Face.
        
        Args:
            split: Dataset split to load (default: "train")
        """
        try:
            logger.info("Loading VQA-RAD dataset from Hugging Face...")
            self.dataset = load_dataset(
                "flaviagiammarino/vqa-rad",
                split=split,
                cache_dir=self.cache_dir
            )
            self._loaded = True

            try:
                dataset_length = len(self.dataset)  # type: ignore
                logger.info(f"Loaded {dataset_length} samples from VQA-RAD dataset")
            except (TypeError, AttributeError):
                logger.info("Loaded VQA-RAD dataset (length not available)")

        except Exception as e:
            logger.error(f"Failed to load VQA-RAD dataset: {e}")
            raise

    def get_sample_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get a sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            Sample data dictionary
        """
        if not self._loaded:
            self.load_dataset()

        if self.dataset is None:
            raise RuntimeError("Dataset failed to load")

        try:
            dataset_length = len(self.dataset)  # type: ignore
        except (TypeError, AttributeError):
            raise RuntimeError("Dataset does not support length operation")

        if index >= dataset_length:
            raise IndexError(f"Index {index} out of range for dataset size {dataset_length}")

        sample = self.dataset[index]  # type: ignore
        return self._process_sample(sample, index)

    def get_sample_by_id(self, sample_id: str) -> Dict[str, Any]:
        """
        Get a sample by ID (assumes ID format is 'sample_XXXX').
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Sample data dictionary
        """
        if not self._loaded:
            self.load_dataset()

        # Extract index from sample_id (format: sample_XXXX)
        try:
            if sample_id.startswith("sample_"):
                index = int(sample_id.split("_")[1])
            else:
                # Try to parse as direct index
                index = int(sample_id)

            return self.get_sample_by_index(index)

        except (ValueError, IndexError) as e:
            logger.error(f"Invalid sample_id {sample_id}: {e}")
            raise ValueError(f"Cannot find sample with ID {sample_id}")

    def _process_sample(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Process a raw dataset sample into the expected format.
        
        Args:
            sample: Raw sample from dataset
            index: Sample index
            
        Returns:
            Processed sample dictionary
        """
        sample_id = f"sample_{index:04d}"

        # Save image to local directory
        image_path = self._save_image(sample["image"], sample_id)

        # Extract question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Determine question type and modality
        question_type = self._classify_question_type(question, answer)
        modality = self._classify_modality(sample.get("image_name", ""))

        return {
            "sample_id": sample_id,
            "image_path": str(image_path),
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "modality": modality,
            "metadata": {
                "original_index": index,
                "image_name": sample.get("image_name", ""),
                "source": "huggingface_vqa_rad",
                "dataset_split": "train"
            }
        }

    def _save_image(self, image: Image.Image, sample_id: str) -> Path:
        """
        Save image to local directory.
        
        Args:
            image: PIL Image object
            sample_id: Sample identifier
            
        Returns:
            Path to saved image
        """
        image_path = self.images_dir / f"{sample_id}.png"

        # Convert to RGB if necessary and save as PNG
        if image.mode in ['RGBA', 'LA']:
            # Handle transparency
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])
            else:
                background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')

        image.save(image_path, 'PNG')
        return image_path

    @staticmethod
    def _classify_question_type(question: str, answer: str) -> str:
        """
        Classify the question type based on question and answer.
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Question type classification
        """
        question_lower = question.lower()
        answer_lower = answer.lower()

        # Yes/No questions
        if answer_lower in ["yes", "no"]:
            return "yes_no"

        # Counting questions
        if any(word in question_lower for word in ["how many", "count", "number of"]):
            return "counting"

        # Color questions
        if "color" in question_lower or "colour" in question_lower:
            return "color"

        # Modality questions
        if any(word in question_lower for word in ["modality", "type of scan", "imaging"]):
            return "modality"

        # Organ/anatomy questions
        if any(word in question_lower for word in ["organ", "anatomy", "part", "what is"]):
            return "organ_identification"

        # Abnormality detection
        if any(word in question_lower for word in ["abnormal", "disease", "condition", "pathology"]):
            return "abnormality_detection"

        # Default classification based on answer length
        if len(answer_lower.split()) == 1:
            return "single_word"
        elif len(answer_lower.split()) <= 3:
            return "short_phrase"
        else:
            return "long_answer"

    @staticmethod
    def _classify_modality(image_name: str) -> str:
        """
        Classify imaging modality based on image name.
        
        Args:
            image_name: Image filename or path
            
        Returns:
            Modality classification
        """
        if not image_name:
            return "unknown"

        image_name_upper = image_name.upper()

        if "CT" in image_name_upper:
            return "CT"
        elif any(term in image_name_upper for term in ["MR", "MRI"]):
            return "MR"
        elif any(term in image_name_upper for term in ["X-RAY", "XRAY", "RADIOGRAPH"]):
            return "X-ray"
        else:
            return "unknown"

    def sample_dataset(
        self,
        n_samples: int = 100,
        balance_by_modality: bool = True,
        balance_by_question_type: bool = True,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Create a balanced sample of the VQA-RAD dataset.
        
        Args:
            n_samples: Number of samples to select
            balance_by_modality: Whether to balance by imaging modality
            balance_by_question_type: Whether to balance by question type
            random_seed: Random seed for reproducibility
            
        Returns:
            Dict containing sample metadata and statistics
        """
        if not self._loaded:
            self.load_dataset()

        if self.dataset is None:
            raise RuntimeError("Dataset failed to load")

        try:
            dataset_length = len(self.dataset)  # type: ignore
        except (TypeError, AttributeError):
            raise RuntimeError("Dataset does not support length operation")

        random.seed(random_seed)

        # Get all samples
        all_samples = []
        for i in range(dataset_length):
            try:
                sample = self._process_sample(self.dataset[i], i)  # type: ignore
                all_samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue

        logger.info(f"Processed {len(all_samples)} samples from dataset")

        # Perform balanced sampling
        if balance_by_modality or balance_by_question_type:
            selected_samples = self._balanced_sample(
                all_samples, n_samples, balance_by_modality, balance_by_question_type
            )
        else:
            selected_samples = random.sample(all_samples, min(n_samples, len(all_samples)))

        # Save sample metadata
        metadata = {
            "sample_count": len(selected_samples),
            "sampling_method": "balanced" if (balance_by_modality or balance_by_question_type) else "random",
            "random_seed": random_seed,
            "dataset_source": "huggingface_vqa_rad",
            "samples": selected_samples,
            "created_timestamp": pd.Timestamp.now().isoformat()
        }

        metadata_file = self.output_dir / "sample_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Created sample dataset with {len(selected_samples)} samples")
        return metadata

    @staticmethod
    def _balanced_sample(
        samples: List[Dict[str, Any]],
        n_samples: int,
        balance_by_modality: bool,
        balance_by_question_type: bool
    ) -> List[Dict[str, Any]]:
        """
        Perform balanced sampling across specified dimensions.
        
        Args:
            samples: List of all available samples
            n_samples: Number of samples to select
            balance_by_modality: Whether to balance by modality
            balance_by_question_type: Whether to balance by question type
            
        Returns:
            List of selected samples
        """
        # Group samples by categories
        groups = {}

        for sample in samples:
            key_parts = []

            if balance_by_modality:
                key_parts.append(sample["modality"])

            if balance_by_question_type:
                key_parts.append(sample["question_type"])

            group_key = "_".join(key_parts) if key_parts else "all"

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(sample)

        # Sample from each group
        samples_per_group = max(1, n_samples // len(groups))
        selected_samples = []

        for group_key, group_samples in groups.items():
            group_sample_size = min(samples_per_group, len(group_samples))
            group_selection = random.sample(group_samples, group_sample_size)
            selected_samples.extend(group_selection)
            logger.info(
                f"Selected {group_sample_size} samples from group '{group_key}' ({len(group_samples)} available)")

        # If we need more samples, randomly select from remaining
        if len(selected_samples) < n_samples:
            remaining = [s for s in samples if s not in selected_samples]
            additional_needed = n_samples - len(selected_samples)
            if remaining:
                additional_samples = random.sample(remaining, min(additional_needed, len(remaining)))
                selected_samples.extend(additional_samples)

        # If we have too many samples, randomly reduce
        if len(selected_samples) > n_samples:
            selected_samples = random.sample(selected_samples, n_samples)

        return selected_samples

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dataset information dictionary
        """
        if not self._loaded:
            self.load_dataset()

        if self.dataset is None:
            raise RuntimeError("Dataset failed to load")

        try:
            dataset_length = len(self.dataset)  # type: ignore
        except (TypeError, AttributeError):
            raise RuntimeError("Dataset does not support length operation")

        # Analyze dataset
        modalities = {}
        question_types = {}

        for i in range(min(100, dataset_length)):  # Sample first 100 for analysis
            try:
                sample = self._process_sample(self.dataset[i], i)  # type: ignore
                modality = sample["modality"]
                question_type = sample["question_type"]

                modalities[modality] = modalities.get(modality, 0) + 1
                question_types[question_type] = question_types.get(question_type, 0) + 1

            except Exception:
                continue

        return {
            "total_samples": dataset_length,
            "modality_distribution": modalities,
            "question_type_distribution": question_types,
            "source": "huggingface_vqa_rad"
        }


def create_huggingface_dataset(
    output_dir: str = "data/vqarad_hf",
    n_samples: int = 300,
    balance_by_modality: bool = True,
    balance_by_question_type: bool = True,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Create a VQA-RAD dataset from Hugging Face for pipeline use.
    
    Args:
        output_dir: Output directory for the dataset
        n_samples: Number of samples to include
        balance_by_modality: Whether to balance by imaging modality
        balance_by_question_type: Whether to balance by question type
        random_seed: Random seed for reproducibility
        
    Returns:
        Dataset metadata
    """
    loader = HuggingFaceVQARADLoader(output_dir=output_dir)

    return loader.sample_dataset(
        n_samples=n_samples,
        balance_by_modality=balance_by_modality,
        balance_by_question_type=balance_by_question_type,
        random_seed=random_seed
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create VQA-RAD dataset from Hugging Face")
    parser.add_argument("--output", "-o", default="data/vqarad_hf", help="Output directory")
    parser.add_argument("--samples", "-n", type=int, default=300, help="Number of samples")
    parser.add_argument("--balance-modality", action="store_true", help="Balance by imaging modality")
    parser.add_argument("--balance-question", action="store_true", help="Balance by question type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create dataset
    result = create_huggingface_dataset(
        output_dir=args.output,
        n_samples=args.samples,
        balance_by_modality=args.balance_modality,
        balance_by_question_type=args.balance_question,
        random_seed=args.seed
    )

    print(f"Created VQA-RAD dataset in {args.output}")
    print(f"Samples: {result.get('sample_count', 0)}")
    print(f"Method: {result.get('sampling_method', 'unknown')}")
