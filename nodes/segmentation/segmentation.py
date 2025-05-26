"""
Gemini Vision Segmentation Node

Performs visual localization using Gemini 2 Flash Vision to identify relevant regions
in medical images for question answering.
"""

import base64
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiVisionSegmenter:
    """
    Visual localization using Gemini Vision for medical image analysis.
    
    Identifies regions of interest based on the medical question and returns
    bounding box coordinates.
    """

    def __init__(self, model: str = "gemini-2.0-flash-exp"):
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
            generate_result = self._call_gemini_vision(image_data, prompt)

            # Parse the response to extract bounding box
            visual_box = self._parse_segmentation_result(generate_result)

            return {
                "visual_box": visual_box,
                "raw_response": generate_result,
                "image_processed": True
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

    @staticmethod
    def _create_segmentation_prompt(text_query: str) -> str:
        """
        Create a prompt for visual localization.
        
        Args:
            text_query: The medical question about the image
            
        Returns:
            Formatted prompt for Gemini Vision
        """
        prompt = f"""
You are a medical imaging AI assistant. Your task is to identify the most relevant region in this medical image that relates to the following question:

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

If no specific region can be identified or the entire image is relevant, you may return a bounding box that covers most or all of the image.
"""
        return prompt

    def _call_gemini_vision(self, image_data: str, prompt: str) -> str:
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

            # Generate content using modern Gemini API
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image_part]
            )

            return response.text if response.text else ""

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise

    def _parse_segmentation_result(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse Gemini's response to extract bounding box information.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Parsed visual box data or None if parsing failed
        """
        try:
            import json
            import re

            # Try to extract JSON from the response
            json_pattern = r'\{.*?\}'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)

            for json_match in json_matches:
                try:
                    parsed = json.loads(json_match)
                    if "bounding_box" in parsed:
                        bbox = parsed["bounding_box"]

                        # Validate bounding box format
                        required_keys = ["x", "y", "width", "height"]
                        if all(key in bbox for key in required_keys):
                            # Ensure coordinates are within valid range
                            bbox = {
                                "x": max(0, min(1, float(bbox["x"]))),
                                "y": max(0, min(1, float(bbox["y"]))),
                                "width": max(0, min(1, float(bbox["width"]))),
                                "height": max(0, min(1, float(bbox["height"]))),
                            }

                            return {
                                "bounding_box": bbox,
                                "confidence": parsed.get("confidence", 0.5),
                                "region_description": parsed.get("region_description", ""),
                                "relevance_reasoning": parsed.get("relevance_reasoning", ""),
                                "format": "normalized_coords"
                            }

                except json.JSONDecodeError:
                    continue

            # If no valid JSON found, try to extract coordinates from text
            logger.warning("Could not parse JSON response, attempting text extraction")
            return self._extract_coords_from_text(response_text)

        except Exception as e:
            logger.error(f"Failed to parse segmentation result: {e}")
            return None

    @staticmethod
    def _extract_coords_from_text(text: str) -> Optional[Dict[str, Any]]:
        """
        Fallback method to extract coordinates from free text response.
        
        Args:
            text: Response text
            
        Returns:
            Extracted visual box data or None
        """
        try:
            import re

            # Look for patterns like "x: 0.2, y: 0.3, width: 0.4, height: 0.5"
            coord_pattern = r'x[:\s]*([0-9.]+).*?y[:\s]*([0-9.]+).*?width[:\s]*([0-9.]+).*?height[:\s]*([0-9.]+)'
            match = re.search(coord_pattern, text, re.IGNORECASE | re.DOTALL)

            if match:
                x, y, width, height = map(float, match.groups())

                # Normalize coordinates
                bbox = {
                    "x": max(0, min(1, x)),
                    "y": max(0, min(1, y)),
                    "width": max(0, min(1, width)),
                    "height": max(0, min(1, height)),
                }

                return {
                    "bounding_box": bbox,
                    "confidence": 0.3,  # Lower confidence for text extraction
                    "region_description": "Extracted from text response",
                    "relevance_reasoning": "Coordinates extracted via text parsing",
                    "format": "normalized_coords"
                }

            # If no coordinates found, return a default box covering the center region
            logger.warning("No coordinates found, using default center region")
            return {
                "bounding_box": {
                    "x": 0.25,
                    "y": 0.25,
                    "width": 0.5,
                    "height": 0.5
                },
                "confidence": 0.1,
                "region_description": "Default center region",
                "relevance_reasoning": "No specific region identified",
                "format": "normalized_coords"
            }

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return None


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
        segmenter = GeminiVisionSegmenter()
        segment_result = segmenter.segment_image(image_path, text_query)

        # Add segmentation metadata
        segment_result["segmenter"] = "GeminiVisionSegmenter"
        segment_result["segmenter_version"] = "v1.0.0"
        segment_result["model"] = "gemini-2.0-flash-exp"

        if segment_result["visual_box"]:
            logger.info("Segmentation completed successfully")
        else:
            logger.warning("Segmentation completed but no visual box identified")

        return segment_result

    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        # Return a fallback result rather than failing completely
        return {
            "visual_box": None,
            "error": str(e),
            "segmenter": "GeminiVisionSegmenter",
            "segmenter_version": "v1.0.0",
            "model": "gemini-2.0-flash-exp",
            "image_processed": False
        }


if __name__ == "__main__":
    # Test the segmentation
    import tempfile
    from PIL import Image
    import numpy as np

    # Create a test image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        # Create the mock medical image
        img_array = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        test_image = Image.fromarray(img_array, mode='L')
        test_image.save(tmp_file.name)

        try:
            result = run_segmentation(
                image_path=tmp_file.name,
                text_query="Where is the heart in this chest X-ray?"
            )

            print("Segmentation test results:")
            print(f"  Visual box found: {result['visual_box'] is not None}")
            if result['visual_box']:
                bbox = result['visual_box']['bounding_box']
                print(
                    f"  Bounding box: x={bbox['x']:.2f}, y={bbox['y']:.2f}, w={bbox['width']:.2f}, h={bbox['height']:.2f}")
                print(f"  Confidence: {result['visual_box']['confidence']:.2f}")
            if 'error' in result:
                print(f"  Error: {result['error']}")

        finally:
            Path(tmp_file.name).unlink()  # Clean up
