"""
Validation Node for MedVoiceQA Pipeline.

Performs quality assessment and critic validation using Gemini models.
"""

import logging
import os
from typing import Dict, Any, Tuple

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiValidationDuo:
    """
    Quality assessment and critic validation using Gemini 2 Flash.
    
    This component acts as a quality gatekeeper, evaluating all pipeline outputs
    and determining if human review is needed.
    """

    def __init__(self, model: str = "gemini-2.0-flash-exp"):
        """Initialize the Gemini validation system."""
        self.model = model
        self.client = None

        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        os.environ['GOOGLE_API_KEY'] = api_key
        self.client = genai.Client()

        # Quality thresholds from registry.json
        self.quality_thresholds = {
            "speech_quality_min": 0.7,
            "uncertainty_max": 0.3,  # High uncertainty triggers review
            "explanation_min_length": 50,
            "visual_box_completeness": True
        }
        # Validation prompt template
        self.validation_prompt = """
You are a medical AI quality assurance specialist. Evaluate this complete medical image analysis pipeline output.

ORIGINAL QUERY: {query}

PIPELINE OUTPUTS:
1. VISUAL LOCALIZATION: {visual_box}
2. SPEECH SYNTHESIS QUALITY: {speech_quality}
3. SPEECH RECOGNITION RESULT: "{asr_text}"
4. MEDICAL REASONING: {explanation}
5. UNCERTAINTY SCORE: {uncertainty}

EVALUATION CRITERIA:
1. VISUAL LOCALIZATION QUALITY:
   - Are the bounding box coordinates reasonable?
   - Does the localization seem relevant to the query?

2. SPEECH PROCESSING QUALITY:
   - Is the speech quality score acceptable (>0.7)?
   - Does the ASR text match the original query?

3. MEDICAL REASONING QUALITY:
   - Is the reasoning medically sound and detailed?
   - Does it properly address the query?
   - Is the explanation clear and comprehensive?

4. CONSISTENCY CHECK:
   - Are all components internally consistent?
   - Do the outputs align with each other?

5. UNCERTAINTY ASSESSMENT:
   - Is the uncertainty score reasonable?
   - High uncertainty (>0.7) may indicate quality issues

PROVIDE ASSESSMENT:
1. NEEDS_HUMAN_REVIEW: [true/false] - Should this sample be flagged for human review?
2. QUALITY_SCORES: Provide scores 0.0-1.0 for:
   - visual_localization_quality
   - speech_processing_quality  
   - reasoning_quality
   - consistency_score
   - overall_quality

3. CRITIC_NOTES: Detailed feedback on any issues found

Format your response as:
NEEDS_REVIEW: [true/false]
VISUAL_LOCALIZATION_QUALITY: [0.0-1.0]
SPEECH_PROCESSING_QUALITY: [0.0-1.0]
REASONING_QUALITY: [0.0-1.0]
CONSISTENCY_SCORE: [0.0-1.0]
OVERALL_QUALITY: [0.0-1.0]
CRITIC_NOTES: [detailed feedback]
"""

    @staticmethod
    def _load_image_for_genai(image_path: str) -> types.Part:
        """Load image as a types.Part for modern Gemini API."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()

            # Determine MIME type
            image_path_lower = image_path.lower()
            if image_path_lower.endswith('.png'):
                mime_type = "image/png"
            elif image_path_lower.endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"
            else:
                # Default to PNG
                mime_type = "image/png"

            return types.Part.from_bytes(data=image_data, mime_type=mime_type)

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

    @staticmethod
    def _parse_validation_response(response_text: str) -> Tuple[bool, str, Dict[str, float]]:
        """Parse validation response into structured data."""
        try:
            lines = response_text.split('\n')
            needs_review = False
            critic_notes = ""
            quality_scores = {}

            for line in lines:
                line = line.strip()
                if line.upper().startswith('NEEDS_REVIEW:'):
                    value = line.split(':', 1)[1].strip().lower()
                    needs_review = value in ['true', '1', 'yes']

                elif line.upper().startswith('VISUAL_LOCALIZATION_QUALITY:'):
                    try:
                        score = float(line.split(':', 1)[1].strip())
                        quality_scores['visual_localization_quality'] = max(0.0, min(1.0, score))
                    except ValueError:
                        quality_scores['visual_localization_quality'] = 0.5

                elif line.upper().startswith('SPEECH_PROCESSING_QUALITY:'):
                    try:
                        score = float(line.split(':', 1)[1].strip())
                        quality_scores['speech_processing_quality'] = max(0.0, min(1.0, score))
                    except ValueError:
                        quality_scores['speech_processing_quality'] = 0.5

                elif line.upper().startswith('REASONING_QUALITY:'):
                    try:
                        score = float(line.split(':', 1)[1].strip())
                        quality_scores['reasoning_quality'] = max(0.0, min(1.0, score))
                    except ValueError:
                        quality_scores['reasoning_quality'] = 0.5

                elif line.upper().startswith('CONSISTENCY_SCORE:'):
                    try:
                        score = float(line.split(':', 1)[1].strip())
                        quality_scores['consistency_score'] = max(0.0, min(1.0, score))
                    except ValueError:
                        quality_scores['consistency_score'] = 0.5

                elif line.upper().startswith('OVERALL_QUALITY:'):
                    try:
                        score = float(line.split(':', 1)[1].strip())
                        quality_scores['overall_quality'] = max(0.0, min(1.0, score))
                    except ValueError:
                        quality_scores['overall_quality'] = 0.5

                elif line.upper().startswith('CRITIC_NOTES:'):
                    critic_notes = line.split(':', 1)[1].strip()

            # Ensure all required scores are present
            required_scores = [
                'visual_localization_quality', 'speech_processing_quality',
                'reasoning_quality', 'consistency_score', 'overall_quality'
            ]
            for score_name in required_scores:
                if score_name not in quality_scores:
                    quality_scores[score_name] = 0.5  # Default medium score

            return needs_review, critic_notes, quality_scores

        except Exception as e:
            logger.error(f"Failed to parse validation response: {e}")
            # Return conservative defaults
            return True, f"Parsing error: {str(e)}", {
                'visual_localization_quality': 0.3,
                'speech_processing_quality': 0.3,
                'reasoning_quality': 0.3,
                'consistency_score': 0.3,
                'overall_quality': 0.3
            }

    def _apply_heuristic_checks(
        self,
        speech_quality_score: float,
        uncertainty: float,
        explanation: str,
        visual_box: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Apply heuristic quality checks."""
        issues = []

        # Check speech quality
        if speech_quality_score < self.quality_thresholds["speech_quality_min"]:
            issues.append(
                f"Low speech quality: {speech_quality_score:.2f} < {self.quality_thresholds['speech_quality_min']}")

        # Check uncertainty
        if uncertainty > self.quality_thresholds["uncertainty_max"]:
            issues.append(f"High uncertainty: {uncertainty:.2f} > {self.quality_thresholds['uncertainty_max']}")

        # Check explanation length
        if len(explanation) < self.quality_thresholds["explanation_min_length"]:
            issues.append(
                f"Short explanation: {len(explanation)} chars < {self.quality_thresholds['explanation_min_length']}")

        # Check visual box completeness
        required_box_keys = ['x1', 'y1', 'x2', 'y2']
        missing_keys = [key for key in required_box_keys if key not in visual_box]
        if missing_keys:
            issues.append(f"Incomplete visual box: missing {missing_keys}")

        needs_review = len(issues) > 0
        notes = "; ".join(issues) if issues else "All heuristic checks passed"

        return needs_review, notes

    def validate_pipeline_output(
        self,
        image_path: str,
        text_query: str,
        visual_box: Dict[str, Any],
        speech_path: str,
        asr_text: str,
        text_explanation: str,
        uncertainty: float,
        speech_quality_score: float
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Perform comprehensive validation of pipeline outputs.
        
        Args:
            image_path: Path to medical image
            text_query: Original query text
            visual_box: Bounding box from segmentation
            speech_path: Path to synthesized speech
            asr_text: ASR transcription result
            text_explanation: Generated medical reasoning
            uncertainty: Uncertainty score
            speech_quality_score: Speech synthesis quality score
            
        Returns:
            Tuple of (needs_review, critic_notes, quality_scores)
        """
        try:
            logger.info("Performing pipeline validation")

            # First apply heuristic checks
            heuristic_review, heuristic_notes = self._apply_heuristic_checks(
                speech_quality_score, uncertainty, text_explanation, visual_box
            )

            # Prepare validation prompt
            prompt = self.validation_prompt.format(
                query=text_query,
                visual_box=str(visual_box),
                speech_quality=speech_quality_score,
                asr_text=asr_text,
                explanation=text_explanation[:500] + "..." if len(text_explanation) > 500 else text_explanation,
                uncertainty=uncertainty
            )
            # Load image for Gemini API
            image_part = self._load_image_for_genai(image_path)

            # Generate validation assessment using modern API
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image_part]
            )
            response_text = response.text.strip()

            # Parse response
            ai_needs_review, ai_notes, quality_scores = self._parse_validation_response(response_text)

            # Combine heuristic and AI assessments
            final_needs_review = heuristic_review or ai_needs_review
            combined_notes = f"Heuristic checks: {heuristic_notes}. AI assessment: {ai_notes}"

            logger.info(f"Validation complete. Needs review: {final_needs_review}")

            return final_needs_review, combined_notes, quality_scores

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            # Conservative fallback - flag for review
            return True, f"Validation error: {str(e)}", {
                'visual_localization_quality': 0.2,
                'speech_processing_quality': 0.2,
                'reasoning_quality': 0.2,
                'consistency_score': 0.2,
                'overall_quality': 0.2
            }


def run_validation(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for pipeline validation.
    
    Args:
        state: Pipeline state containing all outputs from previous nodes
        
    Returns:
        Updated state with needs_review, critic_notes, and quality_scores
    """
    logger.info("Running validation node")

    try:
        # Extract required inputs
        image_path = state.get("image_path")
        text_query = state.get("text_query")
        visual_box = state.get("visual_box")
        speech_path = state.get("speech_path")
        asr_text = state.get("asr_text")
        text_explanation = state.get("text_explanation")
        uncertainty = state.get("uncertainty")
        speech_quality_score = state.get("speech_quality_score")

        # Validate all required inputs are present
        required_inputs = [
            ("image_path", image_path),
            ("text_query", text_query),
            ("visual_box", visual_box),
            ("speech_path", speech_path),
            ("asr_text", asr_text),
            ("text_explanation", text_explanation),
            ("uncertainty", uncertainty),
            ("speech_quality_score", speech_quality_score)
        ]

        missing_inputs = [name for name, value in required_inputs if value is None]
        if missing_inputs:
            logger.error(f"Missing required inputs for validation: {missing_inputs}")
            return {
                **state,
                "needs_review": True,
                "critic_notes": f"Missing required inputs: {missing_inputs}",
                "quality_scores": {
                    'visual_localization_quality': 0.0,
                    'speech_processing_quality': 0.0,
                    'reasoning_quality': 0.0,
                    'consistency_score': 0.0,
                    'overall_quality': 0.0
                }
            }

        # Initialize validation system
        validator = GeminiValidationDuo()

        # Perform validation
        needs_review, critic_notes, quality_scores = validator.validate_pipeline_output(
            image_path=image_path,
            text_query=text_query,
            visual_box=visual_box,
            speech_path=speech_path,
            asr_text=asr_text,
            text_explanation=text_explanation,
            uncertainty=uncertainty,
            speech_quality_score=speech_quality_score
        )

        logger.info(f"Validation complete. Needs review: {needs_review}")

        return {
            **state,
            "needs_review": needs_review,
            "critic_notes": critic_notes,
            "quality_scores": quality_scores
        }

    except Exception as e:
        logger.error(f"Validation node failed: {e}")
        return {
            **state,
            "needs_review": True,
            "critic_notes": f"Validation node error: {str(e)}",
            "quality_scores": {
                'visual_localization_quality': 0.1,
                'speech_processing_quality': 0.1,
                'reasoning_quality': 0.1,
                'consistency_score': 0.1,
                'overall_quality': 0.1
            }
        }
