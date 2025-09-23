"""
Validation Node for MedVoiceQA Pipeline.

Performs quality assessment and critic validation using Gemini models.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from prompts import PromptManager

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Model for validation results with proper field constraints."""
    needs_review: bool
    visual_localization_quality: float = Field(..., ge=0.0, le=1.0)
    speech_processing_quality: float = Field(..., ge=0.0, le=1.0)
    reasoning_quality: float = Field(..., ge=0.0, le=1.0)
    consistency_score: float = Field(..., ge=0.0, le=1.0)
    overall_quality: float = Field(..., ge=0.0, le=1.0)
    critic_notes: str


class GeminiValidationDuo:
    """
    Quality assessment and critic validation using Gemini 2 Flash.
    
    This component acts as a quality gatekeeper, evaluating all pipeline outputs
    and determining if human review is needed.
    """

    def __init__(self, model: str = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')):
        """Initialize the Gemini validation system."""
        self.model = model

        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        os.environ['GOOGLE_API_KEY'] = api_key
        self.client = genai.Client()

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        # Quality thresholds from registry.json
        self.quality_thresholds = {
            "speech_quality_min": 0.7,
            "uncertainty_max": 0.3,  # High uncertainty triggers review
            "explanation_min_length": 50
        }

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
        visual_box_issues = self._validate_visual_box_structure(visual_box)
        if visual_box_issues:
            issues.append(visual_box_issues)

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

            # Get prompt from prompt manager
            default_prompt = """You are a medical AI quality assurance specialist. Evaluate this complete medical image analysis pipeline output.

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

Provide a comprehensive assessment with:
- needs_review: boolean indicating if human review is required
- Quality scores (0.0-1.0) for each component
- Detailed critic notes explaining your assessment

Consider flagging for review if:
- Visual localization seems irrelevant or has poor coordinates
- Speech processing quality is below 0.7
- Medical reasoning is unclear, too brief, or medically unsound
- High uncertainty (>0.7) without clear justification
- Inconsistencies between pipeline components"""

            # Get prompt from prompt manager with validation
            required_variables = {"query", "visual_box", "speech_quality", "asr_text", "explanation", "uncertainty"}
            prompt_template = self.prompt_manager.get_prompt(
                "validation",
                default_prompt,
                required_variables
            )

            # Prepare validation prompt
            prompt = prompt_template.format(
                query=text_query,
                visual_box=str(visual_box),
                speech_quality=speech_quality_score,
                asr_text=asr_text,
                explanation=text_explanation[:500] + "..." if len(text_explanation) > 500 else text_explanation,
                uncertainty=uncertainty
            )
            # Load image for Gemini API
            image_part = self._load_image_for_genai(image_path)

            # Generate validation assessment using structured output
            generation_config = {
                "response_mime_type": "application/json",
                "response_schema": ValidationResult,
            }

            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image_part],
                config=generation_config
            )

            if not response or not response.parsed:
                raise ValueError("No structured response received from Gemini model")

            # Extract validation result from structured response
            validation_result = response.parsed

            # Handle the structured response (could be ValidationResult instance or dict)
            if isinstance(validation_result, ValidationResult):
                ai_needs_review = validation_result.needs_review
                ai_notes = validation_result.critic_notes
                quality_scores = {
                    'visual_localization_quality': validation_result.visual_localization_quality,
                    'speech_processing_quality': validation_result.speech_processing_quality,
                    'reasoning_quality': validation_result.reasoning_quality,
                    'consistency_score': validation_result.consistency_score,
                    'overall_quality': validation_result.overall_quality
                }
            elif isinstance(validation_result, dict):
                ai_needs_review = validation_result.get('needs_review', True)
                ai_notes = validation_result.get('critic_notes', 'No structured response received')
                quality_scores = {
                    'visual_localization_quality': validation_result.get('visual_localization_quality', 0.3),
                    'speech_processing_quality': validation_result.get('speech_processing_quality', 0.3),
                    'reasoning_quality': validation_result.get('reasoning_quality', 0.3),
                    'consistency_score': validation_result.get('consistency_score', 0.3),
                    'overall_quality': validation_result.get('overall_quality', 0.3)
                }
            else:
                raise ValueError(f"Unexpected response type: {type(validation_result)}")

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

    @staticmethod
    def _validate_visual_box_structure(visual_box: Dict[str, Any]) -> str:
        """Validate the structure of the visual box data."""
        if not isinstance(visual_box, dict):
            return "Visual box is not a dictionary"

        # Check for the bounding_box key
        if 'bounding_box' not in visual_box:
            return "Missing 'bounding_box' key in visual_box"

        bounding_box = visual_box['bounding_box']
        if not isinstance(bounding_box, dict):
            return "bounding_box is not a dictionary"

        # Check for required bounding box coordinates
        required_coords = ['x', 'y', 'width', 'height']
        missing_coords = [coord for coord in required_coords if coord not in bounding_box]
        if missing_coords:
            return f"Missing bounding box coordinates: {missing_coords}"

        # Validate coordinate values are numeric and within reasonable ranges
        try:
            x = float(bounding_box['x'])
            y = float(bounding_box['y'])
            width = float(bounding_box['width'])
            height = float(bounding_box['height'])

            # Check if coordinates are reasonable (0-1 for normalized coordinates)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                return f"Bounding box coordinates out of range: x={x}, y={y}, width={width}, height={height}"

        except (ValueError, TypeError):
            return "Bounding box coordinates are not valid numbers"

        # Check for confidence score
        if 'confidence' in visual_box:
            try:
                confidence = float(visual_box['confidence'])
                if not (0 <= confidence <= 1):
                    return f"Confidence score out of range: {confidence}"
            except (ValueError, TypeError):
                return "Confidence score is not a valid number"

        return ""  # No issues found


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
        text_query = state.get("text_query", "")
        visual_box = state.get("visual_box", {})
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
                "validation_failed": True,
                "validation_error": f"Missing required inputs: {missing_inputs}",
                "quality_scores": {
                    'visual_localization_quality': 0.0,
                    'speech_processing_quality': 0.0,
                    'reasoning_quality': 0.0,
                    'consistency_score': 0.0,
                    'overall_quality': 0.0
                }
            }

        # Initialize validation system
        validator = GeminiValidationDuo(model=os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'))

        # Perform validation
        needs_review, critic_notes, quality_scores = validator.validate_pipeline_output(
            image_path=image_path,  # type: ignore
            text_query=text_query,  # type: ignore
            visual_box=visual_box,  # type: ignore
            speech_path=speech_path,  # type: ignore
            asr_text=asr_text,  # type: ignore
            text_explanation=text_explanation,  # type: ignore
            uncertainty=uncertainty,  # type: ignore
            speech_quality_score=speech_quality_score  # type: ignore
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
        raise


if __name__ == "__main__":
    """
    Simple test for run_validation function using data loader.
    """
    import sys
    from pathlib import Path
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Add the data directory to the path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data"))

    from data.huggingface_loader import HuggingFaceVQARADLoader


    def test_run_validation():
        """Simple test to check if run_validation function works."""
        print("=" * 50)
        print("Testing run_validation Function")
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

        # Create mock pipeline state with all required fields
        test_state = {
            "image_path": sample['image_path'],
            "text_query": sample['question'],
            "visual_box": {
                "bounding_box": {
                    "x": 0.15,
                    "y": 0.35,
                    "width": 0.3,
                    "height": 0.35,
                },
                "confidence": 0.85,  # Mock confidence score
                "region_description": "Lung region with potential abnormalities",  # Mock description
                "relevance_reasoning": "This region is relevant to the query as it contains potential abnormalities related to the patient's condition."
            },
            "speech_path": "mock_speech.wav",  # Mock path
            "asr_text": sample['question'],  # Use original question as mock ASR
            "text_explanation": f"This is a mock medical explanation for the query: {sample['question']}. The image shows medical findings that are relevant to the question.",
            "uncertainty": 0.3,
            "speech_quality_score": 0.85
        }

        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  No GOOGLE_API_KEY found - will test error handling")

        # Test run_validation function
        print("3. Testing run_validation...")
        try:
            result = run_validation(test_state)

            # Check result
            expected_fields = ["needs_review", "critic_notes", "quality_scores"]
            success = True

            for field in expected_fields:
                if field not in result:
                    print(f"❌ Missing field: {field}")
                    success = False
                else:
                    value = result[field]
                    print(f"✅ {field}: {type(value).__name__}")

                    if field == "needs_review":
                        print(f"   Value: {value}")
                    elif field == "critic_notes":
                        preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        print(f"   Preview: {preview}")
                    elif field == "quality_scores":
                        if isinstance(value, dict):
                            print(f"   Quality scores:")
                            for score_name, score_value in value.items():
                                print(f"     {score_name}: {score_value:.2f}")
                        else:
                            print(f"   Value: {value}")

            if success:
                print("✅ run_validation function works!")
            else:
                print("❌ run_validation function has issues")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("   This might be expected if GOOGLE_API_KEY is not set or other dependencies missing")

        print("\nTest complete!")


    # Run the test
    test_run_validation()
