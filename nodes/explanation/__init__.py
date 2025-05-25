"""
Explanation node for generating medical reasoning using Gemini 2 Flash Language Model.
Provides detailed explanations and uncertainty quantification for medical VQA tasks.
"""

import asyncio
import logging
import base64
from typing import Dict, Any, Optional
from pathlib import Path

import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)


class GeminiReasoningEngine:
    """
    Generate detailed medical reasoning and uncertainty estimates using Gemini 2 Flash.
    
    This node analyzes medical images, questions, and visual localizations to provide
    comprehensive explanations with uncertainty quantification for medical VQA tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Gemini reasoning engine."""
        self.config = config
        self.api_key = config.get("gemini_api_key")
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Reasoning prompt template
        self.reasoning_prompt = """You are an expert medical AI assistant analyzing medical images and providing detailed reasoning.

Given:
- Medical Image: [Provided as image]
- Question: {question}
- Visual Localization: {visual_box}

Please provide:
1. **Detailed Medical Reasoning**: Step-by-step analysis of the image in relation to the question
2. **Evidence Assessment**: What specific visual features support your reasoning
3. **Uncertainty Analysis**: Rate your confidence and explain any uncertainties
4. **Clinical Context**: Relevant medical knowledge that applies to this case

Requirements:
- Use precise medical terminology
- Explain your reasoning process clearly
- Identify any limitations or uncertainties
- Structure your response for educational value

Format your response as JSON:
{{
    "reasoning_steps": [
        "Step 1: [Description]",
        "Step 2: [Description]",
        ...
    ],
    "visual_evidence": "Detailed description of visual features that support the reasoning",
    "medical_context": "Relevant clinical knowledge and medical background",
    "confidence_assessment": {{
        "overall_confidence": 0.85,
        "reasoning": "Explanation of confidence level",
        "uncertainty_factors": ["Factor 1", "Factor 2", ...]
    }},
    "final_explanation": "Comprehensive explanation combining all aspects"
}}"""

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate medical reasoning and uncertainty estimates.
        
        Args:
            state: Pipeline state containing image_path, text_query, visual_box
            
        Returns:
            Dict containing text_explanation and uncertainty
        """
        try:
            image_path = state.get("image_path")
            text_query = state.get("text_query", "")
            visual_box = state.get("visual_box", {})
            
            if not image_path or not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                return await self._generate_fallback_explanation(text_query)
            
            # Generate reasoning
            explanation_data = await self._generate_reasoning(
                image_path, text_query, visual_box
            )
            
            return {
                "text_explanation": explanation_data["final_explanation"],
                "uncertainty": explanation_data["confidence_assessment"]["overall_confidence"],
                "reasoning_details": explanation_data  # Additional detailed output
            }
            
        except Exception as e:
            logger.error(f"Error in reasoning engine: {e}")
            return await self._generate_fallback_explanation(text_query)
    
    async def _generate_reasoning(
        self, 
        image_path: str, 
        question: str, 
        visual_box: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed medical reasoning using Gemini."""
        try:
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            
            # Format visual localization info
            box_description = self._format_visual_box(visual_box)
            
            # Create prompt
            prompt = self.reasoning_prompt.format(
                question=question,
                visual_box=box_description
            )
            
            # Generate response with retry logic
            for attempt in range(3):
                try:
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        [prompt, image]
                    )
                    
                    # Parse JSON response
                    import json
                    explanation_data = json.loads(response.text.strip())
                    
                    # Validate response structure
                    if self._validate_reasoning_response(explanation_data):
                        return explanation_data
                    else:
                        logger.warning(f"Invalid response structure on attempt {attempt + 1}")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                    if attempt == 2:  # Last attempt
                        return self._create_fallback_reasoning(question, response.text if 'response' in locals() else "")
                except Exception as e:
                    logger.warning(f"API error on attempt {attempt + 1}: {e}")
                    if attempt == 2:
                        raise
                
                await asyncio.sleep(1)  # Brief delay between retries
            
            raise Exception("Failed to generate valid reasoning after 3 attempts")
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return self._create_fallback_reasoning(question)
    
    def _format_visual_box(self, visual_box: Dict[str, Any]) -> str:
        """Format visual localization information for the prompt."""
        if not visual_box:
            return "No specific visual localization provided"
        
        description = []
        if "bounding_boxes" in visual_box:
            for i, box in enumerate(visual_box["bounding_boxes"]):
                box_desc = f"Box {i+1}: {box.get('label', 'Unknown')} at coordinates ({box.get('x', 0)}, {box.get('y', 0)}, {box.get('width', 0)}, {box.get('height', 0)})"
                if box.get("confidence"):
                    box_desc += f" (confidence: {box['confidence']:.2f})"
                description.append(box_desc)
        
        if "key_regions" in visual_box:
            description.append(f"Key regions identified: {', '.join(visual_box['key_regions'])}")
        
        return "; ".join(description) if description else "Visual localization data available but not detailed"
    
    def _validate_reasoning_response(self, data: Dict[str, Any]) -> bool:
        """Validate the structure of the reasoning response."""
        required_fields = [
            "reasoning_steps", "visual_evidence", "medical_context",
            "confidence_assessment", "final_explanation"
        ]
        
        if not all(field in data for field in required_fields):
            return False
        
        # Validate confidence assessment structure
        confidence = data.get("confidence_assessment", {})
        if not isinstance(confidence.get("overall_confidence"), (int, float)):
            return False
        
        if not (0 <= confidence["overall_confidence"] <= 1):
            return False
        
        return True
    
    def _create_fallback_reasoning(self, question: str, raw_response: str = "") -> Dict[str, Any]:
        """Create a fallback reasoning response when normal processing fails."""
        return {
            "reasoning_steps": [
                "Unable to perform detailed visual analysis",
                "Providing general medical knowledge response",
                "Recommending professional medical consultation"
            ],
            "visual_evidence": "Unable to analyze visual features due to processing limitations",
            "medical_context": f"General medical knowledge relevant to: {question}",
            "confidence_assessment": {
                "overall_confidence": 0.3,
                "reasoning": "Low confidence due to processing limitations",
                "uncertainty_factors": ["Processing error", "Unable to analyze image", "Limited information"]
            },
            "final_explanation": raw_response if raw_response else f"I was unable to provide a detailed analysis for the question '{question}' due to processing limitations. Please consult with a medical professional for accurate diagnosis and treatment recommendations.",
            "fallback_used": True
        }
    
    async def _generate_fallback_explanation(self, question: str) -> Dict[str, Any]:
        """Generate a simple fallback explanation."""
        fallback_data = self._create_fallback_reasoning(question)
        return {
            "text_explanation": fallback_data["final_explanation"],
            "uncertainty": fallback_data["confidence_assessment"]["overall_confidence"],
            "reasoning_details": fallback_data
        }
