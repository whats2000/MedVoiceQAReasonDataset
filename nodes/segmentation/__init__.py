"""
Gemini Vision Segmentation Node

Uses Gemini 2 Flash Vision for visual localization and bounding box detection.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class GeminiVisionSegmenter:
    """Visual localization using Gemini 2 Flash Vision"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize the vision model"""
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            max_tokens=2048,
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert medical image analyst specializing in visual localization.
            
Your task is to analyze medical images and provide bounding box coordinates for regions relevant to the given query.

Return your response as a JSON object with the following structure:
{{
  "visual_box": {{
    "x": <x_coordinate>,
    "y": <y_coordinate>, 
    "width": <box_width>,
    "height": <box_height>,
    "confidence": <confidence_score_0_to_1>,
    "description": "<brief_description_of_located_region>"
  }},
  "reasoning": "<explanation_of_localization_decision>"
}}

Coordinates should be normalized to the image dimensions (0.0 to 1.0).
Confidence should reflect your certainty in the localization.
"""),
            ("human", """Please analyze this medical image and locate the region most relevant to this query: "{query}"

Image: {image_path}

Focus on identifying anatomical structures, abnormalities, or regions of interest that directly relate to the question.""")
        ])
    
    async def process(self, image_path: str, text_query: str) -> Dict:
        """
        Process image for visual localization
        
        Args:
            image_path: Path to the medical image
            text_query: Question about the image
            
        Returns:
            Dict containing visual_box coordinates and metadata
        """
        try:
            # Prepare the prompt
            messages = self.prompt_template.format_messages(
                query=text_query,
                image_path=image_path
            )
            
            # Add image to the message
            if Path(image_path).exists():
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                
                # Create message with image
                message_with_image = HumanMessage(
                    content=[
                        {"type": "text", "text": messages[-1].content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._encode_image(image_data)}"
                            }
                        }
                    ]
                )
                
                # Get response from Gemini
                response = await self.llm.ainvoke([messages[0], message_with_image])
                
                # Parse response
                result = self._parse_response(response.content)
                
                logger.info(f"Successfully processed visual localization for: {image_path}")
                return result
            else:
                logger.warning(f"Image file not found: {image_path}")
                return self._create_fallback_result()
                
        except Exception as e:
            logger.error(f"Failed to process visual segmentation: {e}")
            return self._create_fallback_result()
    
    def _encode_image(self, image_data: bytes) -> str:
        """Encode image data to base64"""
        import base64
        return base64.b64encode(image_data).decode('utf-8')
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse Gemini response into structured format"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group())
                
                # Validate and extract visual_box
                visual_box = response_json.get("visual_box", {})
                
                # Ensure all required fields are present
                required_fields = ["x", "y", "width", "height", "confidence"]
                for field in required_fields:
                    if field not in visual_box:
                        visual_box[field] = 0.5 if field != "confidence" else 0.3
                
                # Ensure coordinates are normalized
                for coord in ["x", "y", "width", "height"]:
                    visual_box[coord] = max(0.0, min(1.0, float(visual_box[coord])))
                
                visual_box["confidence"] = max(0.0, min(1.0, float(visual_box["confidence"])))
                
                return {
                    "visual_box": visual_box,
                    "reasoning": response_json.get("reasoning", "Visual localization completed"),
                    "raw_response": response_text
                }
            else:
                logger.warning("Could not parse JSON from Gemini response")
                return self._create_fallback_result()
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return self._create_fallback_result()
    
    def _create_fallback_result(self) -> Dict:
        """Create fallback result when processing fails"""
        return {
            "visual_box": {
                "x": 0.5,
                "y": 0.5,
                "width": 0.3,
                "height": 0.3,
                "confidence": 0.1,
                "description": "Fallback localization - center region"
            },
            "reasoning": "Fallback localization due to processing error",
            "raw_response": "Error: Could not process image"
        }
