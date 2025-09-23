"""
Gemini Vision Segmentation Node

Performs visual localization using Gemini 2 Flash Vision to identify relevant regions
in medical images for question answering.
"""

import base64
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from PIL import Image
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from prompts import PromptManager

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    width: float = Field(..., ge=0, le=1)
    height: float = Field(..., ge=0, le=1)


class SegmentationResult(BaseModel):
    bounding_box: BoundingBox
    confidence: float = Field(..., ge=0, le=1)
    region_description: str
    relevance_reasoning: str


class GeminiVisionSegmenter:
    """
    Visual localization using Gemini Vision for medical image analysis.
    
    Identifies regions of interest based on the medical question and returns
    bounding box coordinates.
    """

    def __init__(self, model: str = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')):
        """
        Initialize the Gemini Vision segmenter.
        
        Args:
            model: Gemini model to use for vision tasks
        """
        self.model = model

        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        os.environ['GOOGLE_API_KEY'] = api_key
        self.client = genai.Client()
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()

    def segment_image(self, image_path: str, text_query: str) -> Dict[str, Any]:
        """
        Perform visual localization on a medical image.
        
        Args:
            image_path: Path to the medical image
            text_query: Medical question/query about the image
            
        Returns:
            Dict containing visual_box coordinates and metadata
        """
        try:
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                return {"visual_box": None, "error": "Image not found"}

            # Prepare the image
            image_data = self._prepare_image(image_path)
            if not image_data:
                return {"visual_box": None, "error": "Failed to prepare image"}

            # Create segmentation prompt
            prompt = self._create_segmentation_prompt(text_query)

            # Call Gemini Vision
            raw_response, visual_box_obj = self._call_gemini_vision(image_data, prompt)

            # Convert only if it's a SegmentationResult
            if isinstance(visual_box_obj, SegmentationResult):
                visual_box = visual_box_obj.model_dump()
            else:
                visual_box = visual_box_obj  # already None or dict

            return {
                "visual_box": visual_box,
                "raw_response": raw_response,
                "image_processed": True,
            }

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return {
                "visual_box": None,
                "error": str(e),
                "image_processed": False
            }

    @staticmethod
    def _prepare_image(image_path: str) -> Optional[str]:
        """
        Prepare image for Gemini Vision API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image data or None if failed
        """
        try:
            # Open and potentially resize image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')

                # Resize if too large (Gemini has size limits)
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Save to memory as JPEG for efficient transmission
                import io
                img_buffer = io.BytesIO()
                if img.mode == 'L':
                    img.save(img_buffer, format='PNG')
                else:
                    img.save(img_buffer, format='JPEG', quality=85)

                # Encode as base64
                img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                return img_data

        except Exception as e:
            logger.error(f"Failed to prepare image {image_path}: {e}")
            return None

    def _create_segmentation_prompt(self, text_query: str) -> str:
        """
        Create a prompt for visual localization.
        
        Args:
            text_query: The medical question about the image
            
        Returns:
            Formatted prompt for Gemini Vision
        """
        # Default prompt template
        default_prompt = """You are a medical imaging AI assistant. Your task is to identify the most relevant region in this medical image that relates to the following question:

Question: {text_query}

Please analyze the image and:
1. Identify the anatomical region or abnormality that is most relevant to answering this question
2. Provide a bounding box (x, y, width, height) in normalized coordinates (0-1) that encompasses this region
3. Explain your reasoning for selecting this region

Your response should be in the following JSON format:
{{
    "bounding_box": {{
        "x": <normalized x coordinate of top-left corner>,
        "y": <normalized y coordinate of top-left corner>, 
        "width": <normalized width>,
        "height": <normalized height>
    }},
    "confidence": <confidence score 0-1>,
    "region_description": "<description of what this region contains>",
    "relevance_reasoning": "<explanation of why this region is relevant to the question>"
}}

If no specific region can be identified or the entire image is relevant, you may return a bounding box that covers most or all of the image."""
        
        # Get prompt from prompt manager with validation
        required_variables = {"text_query"}
        prompt_template = self.prompt_manager.get_prompt(
            "segmentation",
            default_prompt,
            required_variables
        )
        
        # Format the prompt with the text query
        return prompt_template.format(text_query=text_query)

    def _call_gemini_vision(self, image_data: str, prompt: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Call Gemini Vision API with image and prompt.
        
        Args:
            image_data: Base64 encoded image
            prompt: Text prompt for analysis
            
        Returns:
            API response text
        """
        try:
            # Create image part from base64 data
            image_part = types.Part.from_bytes(
                data=base64.b64decode(image_data),
                mime_type="image/jpeg"
            )

            # Any other controls (temperature, top_p…) go here
            generation_config = GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SegmentationResult,
            )

            # Generate content using modern Gemini API
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image_part],
                config=generation_config
            )

            return response.text, response.parsed

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise


def run_segmentation(image_path: str, text_query: str) -> Dict[str, Any]:
    """
    Run the Gemini Vision segmentation node.
    
    Args:
        image_path: Path to the medical image
        text_query: Medical question about the image
        
    Returns:
        Dict containing visual_box and related metadata
        
    Raises:
        Exception: If segmentation fails critically
    """
    logger.info(f"Starting segmentation for image: {Path(image_path).name}")

    try:
        segmenter = GeminiVisionSegmenter(model=os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'))
        segment_result = segmenter.segment_image(image_path, text_query)

        # Add segmentation metadata
        segment_result["segmenter"] = "GeminiVisionSegmenter"
        segment_result["segmenter_version"] = "v1.0.0"
        segment_result["model"] = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')

        if segment_result["visual_box"]:
            logger.info("Segmentation completed successfully")
        else:
            logger.warning("Segmentation completed but no visual box identified")
            # Add failure marker when no visual box is found
            segment_result["segmentation_failed"] = True
            segment_result["segmentation_error"] = segment_result.get("error", "No visual localization found")

        return segment_result

    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        # Return a fallback result rather than failing completely
        return {
            "visual_box": None,
            "error": str(e),
            "segmentation_failed": True,
            "segmentation_error": str(e),
            "segmenter": "GeminiVisionSegmenter",
            "segmenter_version": "v1.0.0",
            "model": os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'),
            "image_processed": False
        }


if __name__ == "__main__":
    """
    Simple test for run_segmentation function using data loader.
    """
    import sys
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Add the data directory to the path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data"))

    from data.huggingface_loader import HuggingFaceVQARADLoader


    def test_run_segmentation():
        """Simple test to check if run_segmentation function works."""
        print("=" * 50)
        print("Testing run_segmentation Function")
        print("=" * 50)

        # Setup logging
        logging.basicConfig(level=logging.INFO)

        # Initialize data loader
        print("1. Loading data...")
        loader = HuggingFaceVQARADLoader(output_dir="data/vqarad_hf")

        # Check if data exists
        data_dir = Path("data/vqarad_hf")
        if not data_dir.exists():
            print("❌ Data directory not found. Please run the data loader first:")
            print("   uv run .\\data\\huggingface_loader.py")
            return

        # Get first sample
        print("2. Getting test sample...")
        try:
            sample = loader.get_sample_by_index(0)
            print(f"✅ Sample loaded: {sample['sample_id']}")
            print(f"   Question: {sample['question']}")
            print(f"   Image: {sample['image_path']}")
        except Exception as e:
            print(f"❌ Error loading sample: {e}")
            return

        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  No GOOGLE_API_KEY found - will test error handling")

        # Test run_segmentation function
        print("3. Testing run_segmentation...")
        try:
            result = run_segmentation(sample['image_path'], sample['question'])

            # Check result
            if "visual_box" in result:
                print("✅ run_segmentation function works!")
                visual_box = result['visual_box']
                if visual_box:
                    print(f"   Visual box found:")
                    if 'bounding_box' in visual_box:
                        bbox = visual_box['bounding_box']
                        print(f"     Coordinates: x={bbox.get('x', 'N/A'):.3f}, y={bbox.get('y', 'N/A'):.3f}")
                        print(f"     Size: w={bbox.get('width', 'N/A'):.3f}, h={bbox.get('height', 'N/A'):.3f}")
                    print(f"     Confidence: {visual_box.get('confidence', 'N/A')}")
                    print(f"     Description: {visual_box.get('region_description', 'N/A')}")
                else:
                    print("   No visual box identified (may be expected if no API key)")

                print(f"   Segmenter: {result.get('segmenter', 'N/A')}")
                print(f"   Version: {result.get('segmenter_version', 'N/A')}")
                print(f"   Image processed: {result.get('image_processed', 'N/A')}")

                if "error" in result:
                    print(f"   Error: {result['error']}")
            else:
                print("❌ Missing 'visual_box' field in result")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("\nTest complete!")


    # Run the test
    test_run_segmentation()
