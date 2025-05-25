"""
VQA-RAD Data Loader Node

Loads VQA-RAD samples and converts DICOM images to PNG format.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional
import logging

import pydicom
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class VQARADLoader:
    """Load and preprocess VQA-RAD dataset samples"""
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize the loader with dataset path"""
        self.data_path = Path(data_path or os.getenv("VQA_RAD_DATA_PATH", "./data/vqa_rad/"))
        
        if not self.data_path.exists():
            logger.warning(f"VQA-RAD data path does not exist: {self.data_path}")
    
    async def process(self, sample_id: str) -> Dict:
        """
        Process a single VQA-RAD sample
        
        Args:
            sample_id: Unique identifier for the sample
            
        Returns:
            Dict containing image_path, text_query, and metadata
        """
        try:
            # Load sample metadata
            metadata = self._load_sample_metadata(sample_id)
            
            # Convert DICOM to PNG if needed
            image_path = await self._ensure_png_image(sample_id, metadata)
            
            # Extract question text
            text_query = metadata.get("question", "")
            
            return {
                "image_path": str(image_path),
                "text_query": text_query,
                "metadata": {
                    "sample_id": sample_id,
                    "original_format": metadata.get("image_format", "unknown"),
                    "modality": metadata.get("modality", "unknown"),
                    "answer": metadata.get("answer", ""),
                    "answer_type": metadata.get("answer_type", "unknown"),
                    "question_type": metadata.get("question_type", "unknown"),
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process sample {sample_id}: {e}")
            raise
    
    def _load_sample_metadata(self, sample_id: str) -> Dict:
        """Load metadata for a specific sample"""
        
        # Try to load from index file
        index_file = self.data_path / "vqa_rad_index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                index = json.load(f)
                
            if sample_id in index:
                return index[sample_id]
        
        # Fallback: create minimal metadata
        logger.warning(f"No metadata found for sample {sample_id}, using defaults")
        return {
            "question": f"What do you see in this medical image? (Sample {sample_id})",
            "answer": "Unknown",
            "image_format": "dicom",
            "modality": "unknown",
            "answer_type": "open",
            "question_type": "what"
        }
    
    async def _ensure_png_image(self, sample_id: str, metadata: Dict) -> Path:
        """Ensure image is in PNG format, convert DICOM if necessary"""
        
        # Check if PNG already exists
        png_path = self.data_path / "images_png" / f"{sample_id}.png"
        if png_path.exists():
            return png_path
        
        # Look for DICOM file
        dicom_path = self._find_dicom_file(sample_id)
        if dicom_path and dicom_path.exists():
            return await self._convert_dicom_to_png(dicom_path, png_path)
        
        # Create placeholder image if no source found
        return self._create_placeholder_image(sample_id, png_path)
    
    def _find_dicom_file(self, sample_id: str) -> Optional[Path]:
        """Find DICOM file for the given sample ID"""
        
        # Common DICOM locations and extensions
        dicom_dirs = ["images", "dicom", "dcm"]
        dicom_exts = [".dcm", ".dicom", ""]
        
        for dicom_dir in dicom_dirs:
            dir_path = self.data_path / dicom_dir
            if not dir_path.exists():
                continue
                
            for ext in dicom_exts:
                dicom_path = dir_path / f"{sample_id}{ext}"
                if dicom_path.exists():
                    return dicom_path
        
        return None
    
    async def _convert_dicom_to_png(self, dicom_path: Path, output_path: Path) -> Path:
        """Convert DICOM image to PNG format"""
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read DICOM file
            dicom_data = pydicom.dcmread(dicom_path)
            
            # Get pixel array
            pixel_array = dicom_data.pixel_array
            
            # Normalize to 0-255 range
            if pixel_array.dtype != np.uint8:
                # Handle different bit depths
                if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                    # Apply windowing if available
                    center = float(dicom_data.WindowCenter)
                    width = float(dicom_data.WindowWidth)
                    low = center - width // 2
                    high = center + width // 2
                    pixel_array = np.clip(pixel_array, low, high)
                    pixel_array = ((pixel_array - low) / (high - low) * 255).astype(np.uint8)
                else:
                    # Simple min-max normalization
                    pixel_array = ((pixel_array - pixel_array.min()) / 
                                 (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image and save as PNG
            image = Image.fromarray(pixel_array)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(output_path, "PNG")
            
            logger.info(f"Converted DICOM to PNG: {dicom_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to convert DICOM {dicom_path}: {e}")
            # Fall back to placeholder
            return self._create_placeholder_image(dicom_path.stem, output_path)
    
    def _create_placeholder_image(self, sample_id: str, output_path: Path) -> Path:
        """Create a placeholder image when source is not available"""
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a simple placeholder image
            placeholder = Image.new('RGB', (512, 512), color='lightgray')
            placeholder.save(output_path, "PNG")
            
            logger.warning(f"Created placeholder image for {sample_id}: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create placeholder for {sample_id}: {e}")
            raise
