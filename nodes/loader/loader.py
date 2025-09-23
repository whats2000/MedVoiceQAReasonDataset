"""
VQA-RAD Loader Node

Loads VQA-RAD samples from Hugging Face dataset and converts DICOM images to PNG format if needed.
Follow the node contract: consumes sample_id, produces image_path, text_query, metadata.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import pydicom
import SimpleITK as sitk
from PIL import Image
import numpy as np

from data.huggingface_loader import HuggingFaceVQARADLoader
from models.workflow import SampleMetadata

logger = logging.getLogger(__name__)


class VQARADLoader:
    """
    VQA-RAD dataset loader with DICOM to PNG conversion capability.
    Uses Hugging Face VQA-RAD dataset as the primary data source.
    """

    def __init__(self, data_path: str = "data/vqarad", output_dir: str = "runs/current", use_huggingface: bool = True):
        """
        Initialize the loader.
        
        Args:
            data_path: Path to VQA-RAD dataset (for local files)
            output_dir: Output directory for converted images
            use_huggingface: Whether to use Hugging Face dataset (default: True)
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_huggingface = use_huggingface

        # Create the images output directory
        self.images_output_dir = self.output_dir / "images"
        self.images_output_dir.mkdir(exist_ok=True)

        # Initialize Hugging Face loader if enabled
        if self.use_huggingface:
            self.hf_loader = HuggingFaceVQARADLoader(
                cache_dir=str(self.data_path / "hf_cache"),
                output_dir=str(self.output_dir / "hf_data")
            )

    def load_sample(
        self,
        sample_id: str,
        image_path: Optional[str] = None,
        text_query: Optional[str] = None,
        metadata: Optional[SampleMetadata] = None
    ) -> Dict[str, Any]:
        """
        Load a VQA-RAD sample by ID or use provided data.
        
        Args:
            sample_id: Unique sample identifier
            image_path: Pre-specified image path (optional)
            text_query: Pre-specified text query (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Dict containing image_path, text_query, and metadata
        """
        try:
            # If image_path and text_query are provided, use them directly
            if image_path and text_query:
                logger.info(f"Using provided data for sample {sample_id}")

                # Ensure image is in correct format
                processed_image_path = self._process_image(image_path, sample_id)

                return {
                    "image_path": str(processed_image_path),
                    "text_query": text_query,
                    "metadata": metadata or {}
                }

            # Try to load from Hugging Face dataset first
            if self.use_huggingface:
                logger.info(f"Loading sample {sample_id} from Hugging Face VQA-RAD dataset")
                try:
                    hf_sample = self.hf_loader.get_sample_by_id(sample_id)

                    # Process the image to ensure the correct format
                    processed_image_path = self._process_image(hf_sample["image_path"], sample_id)

                    return {
                        "image_path": str(processed_image_path),
                        "text_query": hf_sample["question"],
                        "metadata": {
                            **hf_sample["metadata"],
                            **(metadata or {}),
                            "ground_truth_answer": hf_sample["answer"],
                            "question_type": hf_sample["question_type"],
                            "modality": hf_sample["modality"]
                        }
                    }
                except Exception as e:
                    logger.warning(f"Failed to load from Hugging Face dataset: {e}")
                    logger.info("Falling back to local dataset or mock data")

            # Otherwise, try to load from local VQA-RAD dataset
            logger.info(f"Loading sample {sample_id} from local VQA-RAD dataset")
            sample_data = self._load_from_dataset(sample_id)

            return sample_data

        except Exception as e:
            logger.error(f"Failed to load sample {sample_id}: {e}")
            raise

    def _process_image(self, image_path: str, sample_id: str) -> Path:
        """
        Process and convert image to PNG format if needed.
        
        Args:
            image_path: Path to input image
            sample_id: Sample identifier for output naming
            
        Returns:
            Path to processed PNG image
        """
        input_path = Path(image_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Determine output filename
        output_filename = f"{sample_id}.png"
        output_path = self.images_output_dir / output_filename

        # If already PNG and in output directory, return as-is
        if input_path.suffix.lower() == '.png' and input_path.parent == self.images_output_dir:
            return input_path

        try:
            # Handle different image formats
            if input_path.suffix.lower() in ['.dcm', '.dicom']:
                # Convert DICOM to PNG
                self._convert_dicom_to_png(input_path, output_path)
            elif input_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                # Convert standard image formats to PNG
                self._convert_image_to_png(input_path, output_path)
            else:
                # Try to handle as generic image
                logger.warning(f"Unknown image format for {input_path}, attempting generic conversion")
                self._convert_image_to_png(input_path, output_path)

            logger.info(f"Converted image: {input_path} → {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to process image {input_path}: {e}")
            # Fallback: copy original file
            shutil.copy2(input_path, output_path)
            return output_path

    @staticmethod
    def _convert_dicom_to_png(dicom_path: Path, output_path: Path) -> None:
        """
        Convert DICOM image to PNG format.
        
        Args:
            dicom_path: Path to DICOM file
            output_path: Path for output PNG file
        """
        try:
            # Read DICOM using pydicom
            dicom_data = pydicom.dcmread(str(dicom_path))

            # Get pixel data
            pixel_array = dicom_data.pixel_array

            # Normalize to 0-255 range
            if pixel_array.dtype != np.uint8:
                # Handle different DICOM value ranges
                pixel_min = pixel_array.min()
                pixel_max = pixel_array.max()

                if pixel_max > pixel_min:
                    pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
                else:
                    pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

            # Convert to PIL Image and save as PNG
            if len(pixel_array.shape) == 3:
                # Multi-channel image
                image = Image.fromarray(pixel_array)
            else:
                # Grayscale image
                image = Image.fromarray(pixel_array, mode='L')

            image.save(output_path, 'PNG')

        except Exception as e:
            logger.warning(f"Failed to convert DICOM with pydicom, trying SimpleITK: {e}")

            # Fallback to SimpleITK
            try:
                sitk_image = sitk.ReadImage(str(dicom_path))
                sitk_array = sitk.GetArrayFromImage(sitk_image)

                # SimpleITK images are typically (z, y, x) - take middle slice if 3D
                if len(sitk_array.shape) == 3:
                    sitk_array = sitk_array[sitk_array.shape[0] // 2]

                # Normalize
                sitk_array = ((sitk_array - sitk_array.min()) / (sitk_array.max() - sitk_array.min()) * 255).astype(
                    np.uint8)

                image = Image.fromarray(sitk_array, mode='L')
                image.save(output_path, 'PNG')

            except Exception as e2:
                logger.error(f"SimpleITK conversion also failed: {e2}")
                raise e

    @staticmethod
    def _convert_image_to_png(image_path: Path, output_path: Path) -> None:
        """
        Convert standard image formats to PNG.
        
        Args:
            image_path: Path to input image
            output_path: Path for output PNG file
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ['RGBA', 'LA']:
                    # Handle transparency
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')

                img.save(output_path, 'PNG')

        except Exception as e:
            logger.error(f"Failed to convert image {image_path}: {e}")
            raise

    def _load_from_dataset(self, sample_id: str) -> Dict[str, Any]:
        """
        Load sample from VQA-RAD dataset files.
        
        This is a placeholder for actual VQA-RAD dataset loading.
        In practice, this would read from the dataset's JSON files and image directories.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Sample data dict
        """
        # For now, return mock data since we're working with test datasets
        logger.warning(f"VQA-RAD dataset loading not implemented, using mock data for {sample_id}")

        # Create a basic mock image if none exists
        mock_image_path = self.images_output_dir / f"{sample_id}_mock.png"
        if not mock_image_path.exists():
            self._create_mock_medical_image(mock_image_path)

        return {
            "image_path": str(mock_image_path),
            "text_query": f"What is visible in this medical image for sample {sample_id}?",
            "metadata": {
                "source": "mock",
                "sample_id": sample_id,
                "modality": "unknown",
                "created_by_loader": True
            }
        }

    @staticmethod
    def _create_mock_medical_image(output_path: Path) -> None:
        """
        Create a mock medical image for testing purposes.
        
        Args:
            output_path: Path for output image
        """
        try:
            # Create a simple mock medical image (grayscale with some structure)
            # Create 512x512 grayscale image
            img_array = np.random.randint(50, 200, (512, 512), dtype=np.uint8)

            # Add some circular structure to mimic medical imaging
            center_x, center_y = 256, 256
            y, x = np.ogrid[:512, :512]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 100 ** 2
            img_array[mask] = np.clip(img_array[mask] + 50, 0, 255)

            # Add some noise typical of medical images
            noise = np.random.normal(0, 10, (512, 512))
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

            # Save as PNG
            image = Image.fromarray(img_array, mode='L')
            image.save(output_path, 'PNG')

            logger.info(f"Created mock medical image: {output_path}")

        except Exception as e:
            logger.error(f"Failed to create mock image: {e}")
            # Create minimal image as fallback
            image = Image.new('L', (512, 512), color=128)
            image.save(output_path, 'PNG')


def run_loader(
    sample_id: str,
    image_path: Optional[str] = None,
    text_query: Optional[str] = None,
    metadata: Optional[SampleMetadata] = None,
    output_dir: str = "runs/current"
) -> Dict[str, Any]:
    """
    Run the VQA-RAD loader node.
    
    Args:
        sample_id: Unique sample identifier
        image_path: Optional pre-specified image path
        text_query: Optional pre-specified text query
        metadata: Optional additional metadata
        output_dir: Output directory for processed images
        
    Returns:
        Dict containing image_path, text_query, and metadata
        
    Raises:
        Exception: If loading or processing fails
    """
    logger.info(f"Starting loader for sample: {sample_id}")

    try:
        loader = VQARADLoader(output_dir=output_dir)
        load_result = loader.load_sample(
            sample_id=sample_id,
            image_path=image_path,
            text_query=text_query,
            metadata=metadata
        )

        # Add loader metadata
        load_result["metadata"]["loaded_by"] = "VQARADLoader"
        load_result["metadata"]["loader_version"] = "v1.0.0"
        load_result["metadata"]["sample_id"] = sample_id

        logger.info(f"Successfully loaded sample: {sample_id}")
        return load_result

    except Exception as e:
        logger.error(f"Loader failed for sample {sample_id}: {e}")
        raise


if __name__ == "__main__":
    # Test the loader
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        result = run_loader(
            sample_id="test_001",
            output_dir=temp_dir
        )

        print("Loader test results:")
        print(f"  Image path: {result['image_path']}")
        print(f"  Text query: {result['text_query']}")
        print(f"  Metadata: {result['metadata']}")

        # Check if the image exists
        image_path = Path(result['image_path'])
        print(f"  Image exists: {image_path.exists()}")
        if image_path.exists():
            print(f"  Image size: {image_path.stat().st_size} bytes")
