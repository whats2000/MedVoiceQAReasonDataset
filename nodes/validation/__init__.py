"""
Validation node for quality assessment using dual Gemini models as critic and assessor.
Implements quality scoring and determines if human review is needed.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)


class GeminiValidationDuo:
    """
    Quality assessment using dual Gemini models for comprehensive validation.
    
    Uses two Gemini instances: one as an assessor and one as a critic to provide
    thorough quality evaluation and determine if human review is needed.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dual validation system."""
        self.config = config
        self.api_key = config.get("gemini_api_key")
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=self.api_key)
        self.assessor_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.critic_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Quality thresholds
        self.quality_thresholds = {
            "visual_localization_min": 0.7,
            "speech_quality_min": 0.6,
            "reasoning_confidence_min": 0.5,
            "overall_quality_min": 0.65
        }
        
        # Assessment prompt template
        self.assessor_prompt = """You are a medical AI quality assessor. Evaluate the quality of this medical VQA pipeline output.

Input Data:
- Question: {question}
- Visual Localization: {visual_box}
- Generated Explanation: {explanation}
- Speech Quality Score: {speech_quality}
- Reasoning Uncertainty: {uncertainty}

Evaluate these aspects:
1. **Visual Localization Quality**: Are the bounding boxes accurate and relevant?
2. **Explanation Quality**: Is the medical reasoning sound and comprehensive?
3. **Coherence**: Do all components work together logically?
4. **Medical Accuracy**: Is the medical content appropriate and safe?
5. **Completeness**: Does the output fully address the question?

Provide scores (0.0-1.0) and detailed feedback:
{{
    "quality_scores": {{
        "visual_localization": 0.85,
        "explanation_quality": 0.90,
        "medical_accuracy": 0.95,
        "coherence": 0.88,
        "completeness": 0.82,
        "overall": 0.88
    }},
    "detailed_feedback": {{
        "strengths": ["Strength 1", "Strength 2", ...],
        "weaknesses": ["Weakness 1", "Weakness 2", ...],
        "suggestions": ["Suggestion 1", "Suggestion 2", ...]
    }},
    "risk_assessment": {{
        "medical_risk_level": "low|medium|high",
        "safety_concerns": ["Concern 1", "Concern 2", ...],
        "requires_expert_review": true/false
    }}
}}"""

        # Critic prompt template
        self.critic_prompt = """You are a critical medical AI reviewer. Your job is to find potential issues and edge cases.

Review the assessor's evaluation and the original pipeline output. Be thorough and critical:

Original Assessment: {assessment}
Pipeline Output: 
- Question: {question}
- Explanation: {explanation}
- Quality Scores: {quality_scores}

Focus on:
1. **Missed Issues**: What did the assessor overlook?
2. **Edge Cases**: Are there unusual aspects that need attention?
3. **Medical Safety**: Any potential safety concerns?
4. **False Confidence**: Is the system overconfident in any area?
5. **Bias Detection**: Any potential biases in the reasoning?

Provide critical analysis:
{{
    "critic_assessment": {{
        "agrees_with_assessor": true/false,
        "additional_concerns": ["Concern 1", "Concern 2", ...],
        "severity_level": "low|medium|high",
        "human_review_recommended": true/false
    }},
    "specific_issues": {{
        "visual_analysis": ["Issue 1", "Issue 2", ...],
        "reasoning_flaws": ["Flaw 1", "Flaw 2", ...],
        "medical_concerns": ["Concern 1", "Concern 2", ...],
        "technical_issues": ["Issue 1", "Issue 2", ...]
    }},
    "recommendation": {{
        "action": "approve|review|reject",
        "priority": "low|medium|high|urgent",
        "notes": "Detailed recommendation"
    }}
}}"""

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform dual validation assessment.
        
        Args:
            state: Pipeline state with all processing results
            
        Returns:
            Dict containing needs_review, critic_notes, quality_scores
        """
        try:
            # Extract required data
            image_path = state.get("image_path")
            text_query = state.get("text_query", "")
            visual_box = state.get("visual_box", {})
            speech_path = state.get("speech_path", "")
            asr_text = state.get("asr_text", "")
            text_explanation = state.get("text_explanation", "")
            uncertainty = state.get("uncertainty", 1.0)
            speech_quality_score = state.get("speech_quality_score", 0.0)
            
            # Run assessor evaluation
            assessor_result = await self._run_assessor(
                text_query, visual_box, text_explanation, 
                speech_quality_score, uncertainty
            )
            
            # Run critic evaluation
            critic_result = await self._run_critic(
                text_query, text_explanation, assessor_result
            )
            
            # Determine if human review is needed
            needs_review = self._determine_review_need(assessor_result, critic_result)
            
            # Compile final results
            return {
                "needs_review": needs_review,
                "critic_notes": self._compile_critic_notes(critic_result),
                "quality_scores": assessor_result.get("quality_scores", {}),
                "assessor_feedback": assessor_result.get("detailed_feedback", {}),
                "critic_assessment": critic_result.get("critic_assessment", {}),
                "validation_metadata": {
                    "assessor_model": "gemini-2.0-flash-exp",
                    "critic_model": "gemini-2.0-flash-exp",
                    "validation_timestamp": self._get_timestamp()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in validation duo: {e}")
            return self._generate_fallback_validation()
    
    async def _run_assessor(
        self, 
        question: str, 
        visual_box: Dict[str, Any], 
        explanation: str,
        speech_quality: float,
        uncertainty: float
    ) -> Dict[str, Any]:
        """Run the assessor model evaluation."""
        try:
            prompt = self.assessor_prompt.format(
                question=question,
                visual_box=json.dumps(visual_box, indent=2),
                explanation=explanation,
                speech_quality=speech_quality,
                uncertainty=uncertainty
            )
            
            for attempt in range(3):
                try:
                    response = await asyncio.to_thread(
                        self.assessor_model.generate_content,
                        prompt
                    )
                    
                    result = json.loads(response.text.strip())
                    if self._validate_assessor_response(result):
                        return result
                    
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Assessor attempt {attempt + 1} failed: {e}")
                    if attempt == 2:
                        return self._create_fallback_assessment()
                
                await asyncio.sleep(1)
            
            return self._create_fallback_assessment()
            
        except Exception as e:
            logger.error(f"Error in assessor: {e}")
            return self._create_fallback_assessment()
    
    async def _run_critic(
        self, 
        question: str, 
        explanation: str, 
        assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the critic model evaluation."""
        try:
            prompt = self.critic_prompt.format(
                assessment=json.dumps(assessment, indent=2),
                question=question,
                explanation=explanation,
                quality_scores=json.dumps(assessment.get("quality_scores", {}), indent=2)
            )
            
            for attempt in range(3):
                try:
                    response = await asyncio.to_thread(
                        self.critic_model.generate_content,
                        prompt
                    )
                    
                    result = json.loads(response.text.strip())
                    if self._validate_critic_response(result):
                        return result
                    
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Critic attempt {attempt + 1} failed: {e}")
                    if attempt == 2:
                        return self._create_fallback_critic()
                
                await asyncio.sleep(1)
            
            return self._create_fallback_critic()
            
        except Exception as e:
            logger.error(f"Error in critic: {e}")
            return self._create_fallback_critic()
    
    def _determine_review_need(
        self, 
        assessor_result: Dict[str, Any], 
        critic_result: Dict[str, Any]
    ) -> bool:
        """Determine if human review is needed based on both assessments."""
        try:
            # Check quality thresholds
            quality_scores = assessor_result.get("quality_scores", {})
            overall_quality = quality_scores.get("overall", 0.0)
            
            # Check individual quality metrics
            visual_quality = quality_scores.get("visual_localization", 0.0)
            medical_accuracy = quality_scores.get("medical_accuracy", 0.0)
            
            # Check risk assessment
            risk_assessment = assessor_result.get("risk_assessment", {})
            high_risk = risk_assessment.get("medical_risk_level") == "high"
            expert_review_needed = risk_assessment.get("requires_expert_review", False)
            
            # Check critic recommendations
            critic_assessment = critic_result.get("critic_assessment", {})
            critic_recommends_review = critic_assessment.get("human_review_recommended", False)
            high_severity = critic_assessment.get("severity_level") == "high"
            
            # Check recommendation action
            recommendation = critic_result.get("recommendation", {})
            action_needs_review = recommendation.get("action") in ["review", "reject"]
            urgent_priority = recommendation.get("priority") == "urgent"
            
            # Decision logic
            needs_review = (
                overall_quality < self.quality_thresholds["overall_quality_min"] or
                visual_quality < self.quality_thresholds["visual_localization_min"] or
                medical_accuracy < 0.8 or  # Higher threshold for medical accuracy
                high_risk or
                expert_review_needed or
                critic_recommends_review or
                high_severity or
                action_needs_review or
                urgent_priority
            )
            
            return needs_review
            
        except Exception as e:
            logger.error(f"Error determining review need: {e}")
            return True  # Default to requiring review on error
    
    def _compile_critic_notes(self, critic_result: Dict[str, Any]) -> str:
        """Compile critic feedback into readable notes."""
        try:
            notes = []
            
            # Add critic assessment summary
            critic_assessment = critic_result.get("critic_assessment", {})
            if not critic_assessment.get("agrees_with_assessor", True):
                notes.append("CRITIC DISAGREEMENT: The critic disagrees with the initial assessment.")
            
            # Add specific concerns
            concerns = critic_assessment.get("additional_concerns", [])
            if concerns:
                notes.append(f"Additional Concerns: {'; '.join(concerns)}")
            
            # Add specific issues
            specific_issues = critic_result.get("specific_issues", {})
            for category, issues in specific_issues.items():
                if issues:
                    notes.append(f"{category.replace('_', ' ').title()}: {'; '.join(issues)}")
            
            # Add recommendation
            recommendation = critic_result.get("recommendation", {})
            if recommendation.get("notes"):
                notes.append(f"Recommendation: {recommendation['notes']}")
            
            return "\n".join(notes) if notes else "No specific concerns identified."
            
        except Exception as e:
            logger.error(f"Error compiling critic notes: {e}")
            return "Error compiling critic feedback."
    
    def _validate_assessor_response(self, data: Dict[str, Any]) -> bool:
        """Validate assessor response structure."""
        required_fields = ["quality_scores", "detailed_feedback", "risk_assessment"]
        return all(field in data for field in required_fields)
    
    def _validate_critic_response(self, data: Dict[str, Any]) -> bool:
        """Validate critic response structure."""
        required_fields = ["critic_assessment", "specific_issues", "recommendation"]
        return all(field in data for field in required_fields)
    
    def _create_fallback_assessment(self) -> Dict[str, Any]:
        """Create fallback assessment when normal processing fails."""
        return {
            "quality_scores": {
                "visual_localization": 0.5,
                "explanation_quality": 0.5,
                "medical_accuracy": 0.5,
                "coherence": 0.5,
                "completeness": 0.5,
                "overall": 0.5
            },
            "detailed_feedback": {
                "strengths": ["Processing completed"],
                "weaknesses": ["Assessment failed - using fallback"],
                "suggestions": ["Manual review recommended"]
            },
            "risk_assessment": {
                "medical_risk_level": "medium",
                "safety_concerns": ["Assessment failure"],
                "requires_expert_review": True
            },
            "fallback_used": True
        }
    
    def _create_fallback_critic(self) -> Dict[str, Any]:
        """Create fallback critic assessment."""
        return {
            "critic_assessment": {
                "agrees_with_assessor": False,
                "additional_concerns": ["Critic assessment failed"],
                "severity_level": "medium",
                "human_review_recommended": True
            },
            "specific_issues": {
                "technical_issues": ["Critic processing failed"]
            },
            "recommendation": {
                "action": "review",
                "priority": "medium",
                "notes": "Critic assessment failed - manual review recommended"
            },
            "fallback_used": True
        }
    
    def _generate_fallback_validation(self) -> Dict[str, Any]:
        """Generate fallback validation result."""
        return {
            "needs_review": True,
            "critic_notes": "Validation system failed - manual review required",
            "quality_scores": {
                "overall": 0.3,
                "validation_failed": True
            },
            "validation_metadata": {
                "error": "Validation system failure",
                "fallback_used": True,
                "validation_timestamp": self._get_timestamp()
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
