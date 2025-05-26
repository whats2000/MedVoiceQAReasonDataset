"""
Explanation Node for MedVoiceQA Pipeline.

Generates reasoning and uncertainty estimates using Gemini 2 Flash Language model.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiReasoningEngine:
    """
    Generates detailed medical reasoning and uncertainty estimates using Gemini 2 Flash.
    
    This component analyzes medical images, queries, and visual localization data
    to produce explainable reasoning chains and uncertainty quantification.
    """

    def __init__(self, model: str = "gemini-2.0-flash-exp"):
        """Initialize the Gemini reasoning engine."""
        self.model = model

        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        os.environ['GOOGLE_API_KEY'] = api_key
        self.client = genai.Client()

        # Reasoning prompt template
        self.reasoning_prompt = """
You are an expert medical imaging specialist. Analyze this medical image and provide detailed reasoning.

MEDICAL IMAGE QUERY: {query}

VISUAL LOCALIZATION: The relevant region has been identified with the following bounding box:
- Coordinates: {visual_box}

Please provide:

1. DETAILED MEDICAL REASONING:
   - What anatomical structures are visible?
   - What abnormalities or findings do you observe?
   - How does the visual localization relate to the query?
   - What differential diagnoses should be considered?
   - What is your final assessment?

2. REASONING CHAIN:
   - Step 1: Initial observation
   - Step 2: Feature analysis
   - Step 3: Clinical correlation
   - Step 4: Conclusion

3. UNCERTAINTY ASSESSMENT:
   - How confident are you in this assessment? (0.0 = very uncertain, 1.0 = very certain)
   - What factors contribute to uncertainty?
   - What additional information would improve confidence?

Format your response as:
REASONING: [detailed medical reasoning]
UNCERTAINTY_SCORE: [float between 0.0 and 1.0]
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
    def _extract_uncertainty_score(reasoning_text: str) -> float:
        """Extract uncertainty score from reasoning text."""
        try:
            # Look for UNCERTAINTY_SCORE: pattern
            lines = reasoning_text.split('\n')
            for line in lines:
                if 'UNCERTAINTY_SCORE:' in line.upper():
                    score_text = line.split(':')[-1].strip()
                    # Extract first float found
                    import re
                    numbers = re.findall(r'(\d+\.?\d*)', score_text)
                    if numbers:
                        score = float(numbers[0])
                        # Ensure the score is between 0 and 1
                        return max(0.0, min(1.0, score))

            # Fallback: try to estimate uncertainty from language
            uncertain_words = ['uncertain', 'unclear', 'difficult', 'challenging', 'ambiguous']
            confident_words = ['clear', 'obvious', 'definite', 'certain', 'confident']

            text_lower = reasoning_text.lower()
            uncertain_count = sum(1 for word in uncertain_words if word in text_lower)
            confident_count = sum(1 for word in confident_words if word in text_lower)

            if confident_count > uncertain_count:
                return 0.8  # High confidence
            elif uncertain_count > confident_count:
                return 0.4  # Low confidence
            else:
                return 0.6  # Medium confidence

        except Exception as e:
            logger.warning(f"Failed to extract uncertainty score: {e}")
            return 0.5  # Default medium uncertainty

    def generate_reasoning(
        self,
        image_path: str,
        text_query: str,
        visual_box: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Generate detailed medical reasoning and uncertainty estimate.
        
        Args:
            image_path: Path to the medical image
            text_query: Medical question/query
            visual_box: Bounding box coordinates from segmentation
            
        Returns:
            Tuple of (reasoning_text, uncertainty_score)
        """
        try:
            logger.info(f"Generating reasoning for query: {text_query}")

            # Format visual box information
            box_str = f"x1={visual_box.get('x1', 'N/A')}, y1={visual_box.get('y1', 'N/A')}, x2={visual_box.get('x2', 'N/A')}, y2={visual_box.get('y2', 'N/A')}"

            # Prepare prompt
            prompt = self.reasoning_prompt.format(
                query=text_query,
                visual_box=box_str
            )

            # Load image for Gemini API
            image_part = self._load_image_for_genai(image_path)

            # Generate reasoning using modern API
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image_part]
            )

            if not response or not response.text:
                raise ValueError("No response text received from Gemini model")

            reasoning_text = response.text.strip()

            # Extract uncertainty score
            uncertainty = self._extract_uncertainty_score(reasoning_text)

            # Clean reasoning text (remove uncertainty score line)
            reasoning_lines = reasoning_text.split('\n')
            cleaned_lines = [
                line for line in reasoning_lines
                if not line.upper().startswith('UNCERTAINTY_SCORE:')
            ]
            clean_reasoning = '\n'.join(cleaned_lines).strip()

            logger.info(f"Generated reasoning (uncertainty: {uncertainty:.2f})")
            return clean_reasoning, uncertainty

        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            # Return fallback reasoning
            fallback_reasoning = f"Unable to generate detailed reasoning for query: {text_query}. Error: {str(e)}"
            return fallback_reasoning, 0.1  # High uncertainty for failures


def run_explanation(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for explanation generation.
    
    Args:
        state: Pipeline state containing image_path, text_query, and visual_box
        
    Returns:
        Updated state with text_explanation and uncertainty
    """
    logger.info("Running explanation node")

    try:
        # Validate inputs
        image_path = state.get("image_path")
        text_query = state.get("text_query")
        visual_box = state.get("visual_box")

        if not image_path or not text_query or not visual_box:
            logger.error("Missing required inputs for explanation node")
            return {
                **state,
                "text_explanation": "Error: Missing required inputs for reasoning generation",
                "uncertainty": 0.1
            }

        # Check if the image file exists
        if not Path(image_path).exists():
            logger.error(f"Image file not found: {image_path}")
            return {
                **state,
                "text_explanation": f"Error: Image file not found: {image_path}",
                "uncertainty": 0.1
            }

        # Initialize reasoning engine
        reasoning_engine = GeminiReasoningEngine()

        # Generate reasoning
        explanation, uncertainty = reasoning_engine.generate_reasoning(
            image_path=image_path,
            text_query=text_query,
            visual_box=visual_box
        )

        logger.info(f"Generated explanation with uncertainty: {uncertainty:.2f}")

        return {
            **state,
            "text_explanation": explanation,
            "uncertainty": uncertainty
        }

    except Exception as e:
        logger.error(f"Explanation node failed: {e}")
        return {
            **state,
            "text_explanation": f"Error in explanation generation: {str(e)}",
            "uncertainty": 0.1
        }


if __name__ == "__main__":
    """
    Simple test for run_explanation function using data loader.
    """
    import sys
    import os
    import logging
    from pathlib import Path
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Add the data directory to the path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data"))

    from data.huggingface_loader import HuggingFaceVQARADLoader


    def test_run_explanation():
        """Simple test to check if run_explanation function works."""
        print("=" * 50)
        print("Testing run_explanation Function")
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

        # Create test state
        test_state = {
            "image_path": sample['image_path'],
            "text_query": sample['question'],
            "visual_box": {
                "x1": 100,
                "y1": 100,
                "x2": 400,
                "y2": 400,
                "confidence": 0.8
            }
        }

        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  No GOOGLE_API_KEY found - will test error handling")

        # Test run_explanation function
        print("3. Testing run_explanation...")
        try:
            result = run_explanation(test_state)

            # Check result
            if "text_explanation" in result and "uncertainty" in result:
                print("✅ run_explanation function works!")
                print(f"   Explanation: {result['text_explanation'][:100]}...")
                print(f"   Uncertainty: {result['uncertainty']}")
            else:
                print("❌ Missing expected fields in result")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("\nTest complete!")


    # Run the test
    test_run_explanation()
