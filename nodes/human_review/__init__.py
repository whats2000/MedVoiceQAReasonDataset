"""
Human-in-the-loop review node for manual quality assessment and corrections.
Provides interface for human reviewers to validate and improve pipeline outputs.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class HumanReviewNode:
    """
    Human-in-the-loop review system for pipeline quality assurance.
    
    Provides structured interface for human reviewers to assess pipeline outputs,
    make corrections, and provide feedback for system improvement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the human review system."""
        self.config = config
        self.review_mode = config.get("review_mode", "interactive")  # interactive, batch, or automatic
        self.review_timeout = config.get("review_timeout", 300)  # 5 minutes default
        self.auto_approve_threshold = config.get("auto_approve_threshold", 0.9)
        
        # Setup review storage
        self.review_dir = Path(config.get("review_dir", "data/reviews"))
        self.review_dir.mkdir(parents=True, exist_ok=True)
        
        # Review categories
        self.review_categories = {
            "visual_localization": "Accuracy of bounding boxes and visual annotations",
            "medical_reasoning": "Correctness and completeness of medical explanations",
            "speech_quality": "Quality of text-to-speech and speech-to-text conversion",
            "overall_coherence": "How well all components work together",
            "safety_assessment": "Medical safety and appropriateness of responses"
        }
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process human review request.
        
        Args:
            state: Complete pipeline state with all processing results
            
        Returns:
            Dict containing review_status and review_notes
        """
        try:
            sample_data = self._extract_sample_data(state)
            critic_notes = state.get("critic_notes", "")
            quality_scores = state.get("quality_scores", {})
            
            # Check if automatic approval is possible
            if self._can_auto_approve(quality_scores, critic_notes):
                return await self._auto_approve(sample_data)
            
            # Determine review mode
            if self.review_mode == "interactive":
                return await self._interactive_review(sample_data, critic_notes)
            elif self.review_mode == "batch":
                return await self._batch_review(sample_data, critic_notes)
            else:  # automatic fallback
                return await self._automatic_review(sample_data, critic_notes)
            
        except Exception as e:
            logger.error(f"Error in human review: {e}")
            return self._generate_error_response(str(e))
    
    def _extract_sample_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize sample data for review."""
        return {
            "sample_id": state.get("sample_id", "unknown"),
            "image_path": state.get("image_path", ""),
            "text_query": state.get("text_query", ""),
            "visual_box": state.get("visual_box", {}),
            "speech_path": state.get("speech_path", ""),
            "asr_text": state.get("asr_text", ""),
            "text_explanation": state.get("text_explanation", ""),
            "uncertainty": state.get("uncertainty", 1.0),
            "speech_quality_score": state.get("speech_quality_score", 0.0),
            "quality_scores": state.get("quality_scores", {}),
            "reasoning_details": state.get("reasoning_details", {}),
            "validation_metadata": state.get("validation_metadata", {})
        }
    
    def _can_auto_approve(self, quality_scores: Dict[str, Any], critic_notes: str) -> bool:
        """Check if sample can be automatically approved without human review."""
        try:
            overall_quality = quality_scores.get("overall", 0.0)
            medical_accuracy = quality_scores.get("medical_accuracy", 0.0)
            
            # High quality thresholds for auto-approval
            high_quality = (
                overall_quality >= self.auto_approve_threshold and
                medical_accuracy >= 0.95
            )
            
            # No critical concerns in critic notes
            critical_keywords = ["high risk", "urgent", "safety concern", "medical error"]
            no_critical_concerns = not any(keyword in critic_notes.lower() for keyword in critical_keywords)
            
            return high_quality and no_critical_concerns
            
        except Exception as e:
            logger.error(f"Error in auto-approval check: {e}")
            return False
    
    async def _auto_approve(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically approve high-quality samples."""
        review_record = {
            "review_type": "auto_approved",
            "timestamp": datetime.now().isoformat(),
            "reviewer": "system",
            "sample_id": sample_data.get("sample_id"),
            "decision": "approved",
            "review_notes": "Automatically approved due to high quality scores and no critical concerns.",
            "quality_assessment": "High quality output meeting auto-approval criteria."
        }
        
        # Save review record
        await self._save_review_record(sample_data.get("sample_id"), review_record)
        
        return {
            "review_status": "approved",
            "review_notes": review_record["review_notes"],
            "reviewer": "system",
            "review_metadata": review_record
        }
    
    async def _interactive_review(self, sample_data: Dict[str, Any], critic_notes: str) -> Dict[str, Any]:
        """Conduct interactive human review."""
        try:
            # Create review interface data
            review_interface = self._create_review_interface(sample_data, critic_notes)
            
            # Save review request
            review_id = f"review_{sample_data.get('sample_id')}_{int(datetime.now().timestamp())}"
            review_file = self.review_dir / f"{review_id}.json"
            
            with open(review_file, 'w') as f:
                json.dump(review_interface, f, indent=2)
            
            logger.info(f"Review request created: {review_file}")
            
            # In a real implementation, this would:
            # 1. Send notification to human reviewers
            # 2. Wait for review completion
            # 3. Load and return review results
            
            # For now, simulate review process or queue for later
            return await self._simulate_review_process(sample_data, review_id)
            
        except Exception as e:
            logger.error(f"Error in interactive review: {e}")
            return self._generate_error_response(str(e))
    
    async def _batch_review(self, sample_data: Dict[str, Any], critic_notes: str) -> Dict[str, Any]:
        """Add sample to batch review queue."""
        try:
            batch_file = self.review_dir / "batch_queue.json"
            
            # Load existing batch queue
            batch_queue = []
            if batch_file.exists():
                with open(batch_file, 'r') as f:
                    batch_queue = json.load(f)
            
            # Add new review item
            review_item = {
                "sample_id": sample_data.get("sample_id"),
                "timestamp": datetime.now().isoformat(),
                "sample_data": sample_data,
                "critic_notes": critic_notes,
                "status": "pending"
            }
            
            batch_queue.append(review_item)
            
            # Save updated queue
            with open(batch_file, 'w') as f:
                json.dump(batch_queue, f, indent=2)
            
            logger.info(f"Sample {sample_data.get('sample_id')} added to batch review queue")
            
            return {
                "review_status": "queued",
                "review_notes": f"Sample queued for batch review. Queue position: {len(batch_queue)}",
                "reviewer": "batch_system",
                "review_metadata": {
                    "queue_position": len(batch_queue),
                    "queued_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in batch review: {e}")
            return self._generate_error_response(str(e))
    
    async def _automatic_review(self, sample_data: Dict[str, Any], critic_notes: str) -> Dict[str, Any]:
        """Perform automatic review based on predefined rules."""
        try:
            quality_scores = sample_data.get("quality_scores", {})
            overall_quality = quality_scores.get("overall", 0.0)
            
            # Simple rule-based automatic review
            if overall_quality >= 0.8:
                decision = "approved"
                notes = "Automatic approval based on quality thresholds."
            elif overall_quality >= 0.6:
                decision = "approved_with_notes"
                notes = f"Conditionally approved. Quality score: {overall_quality:.2f}. Monitor for issues."
            else:
                decision = "rejected"
                notes = f"Automatic rejection due to low quality score: {overall_quality:.2f}"
            
            # Factor in critic concerns
            if "high risk" in critic_notes.lower() or "safety concern" in critic_notes.lower():
                decision = "rejected"
                notes += " Additional rejection due to safety concerns identified by critic."
            
            review_record = {
                "review_type": "automatic",
                "timestamp": datetime.now().isoformat(),
                "reviewer": "automatic_system",
                "sample_id": sample_data.get("sample_id"),
                "decision": decision,
                "review_notes": notes,
                "quality_assessment": f"Overall quality: {overall_quality:.2f}",
                "critic_summary": critic_notes[:200] + "..." if len(critic_notes) > 200 else critic_notes
            }
            
            # Save review record
            await self._save_review_record(sample_data.get("sample_id"), review_record)
            
            return {
                "review_status": decision,
                "review_notes": notes,
                "reviewer": "automatic_system",
                "review_metadata": review_record
            }
            
        except Exception as e:
            logger.error(f"Error in automatic review: {e}")
            return self._generate_error_response(str(e))
    
    def _create_review_interface(self, sample_data: Dict[str, Any], critic_notes: str) -> Dict[str, Any]:
        """Create structured review interface for human reviewers."""
        return {
            "review_metadata": {
                "sample_id": sample_data.get("sample_id"),
                "created_timestamp": datetime.now().isoformat(),
                "review_version": "1.0",
                "requires_review": True
            },
            "sample_overview": {
                "image_path": sample_data.get("image_path"),
                "original_question": sample_data.get("text_query"),
                "generated_explanation": sample_data.get("text_explanation"),
                "confidence_level": sample_data.get("uncertainty", 1.0)
            },
            "processing_results": {
                "visual_localization": sample_data.get("visual_box"),
                "speech_synthesis_path": sample_data.get("speech_path"),
                "speech_recognition_text": sample_data.get("asr_text"),
                "quality_scores": sample_data.get("quality_scores"),
                "reasoning_details": sample_data.get("reasoning_details")
            },
            "validation_feedback": {
                "critic_notes": critic_notes,
                "quality_assessment": sample_data.get("quality_scores", {}),
                "concerns_identified": self._extract_concerns(critic_notes)
            },
            "review_categories": self.review_categories,
            "review_instructions": {
                "task": "Please review the pipeline output and provide assessment",
                "focus_areas": [
                    "Verify visual localization accuracy",
                    "Assess medical reasoning quality",
                    "Check speech processing quality",
                    "Evaluate overall coherence",
                    "Identify any safety concerns"
                ],
                "response_format": {
                    "decision": "approved|approved_with_notes|needs_revision|rejected",
                    "category_scores": "1-5 scale for each review category",
                    "feedback": "Detailed written feedback",
                    "corrections": "Any necessary corrections or improvements",
                    "priority": "low|medium|high|urgent"
                }
            }
        }
    
    def _extract_concerns(self, critic_notes: str) -> List[str]:
        """Extract key concerns from critic notes."""
        concerns = []
        concern_keywords = {
            "medical accuracy": ["medical error", "diagnosis", "clinical"],
            "safety": ["safety", "risk", "harm", "dangerous"],
            "technical": ["processing", "error", "failure", "technical"],
            "quality": ["low quality", "poor", "inadequate", "insufficient"]
        }
        
        for category, keywords in concern_keywords.items():
            if any(keyword in critic_notes.lower() for keyword in keywords):
                concerns.append(category)
        
        return concerns
    
    async def _simulate_review_process(self, sample_data: Dict[str, Any], review_id: str) -> Dict[str, Any]:
        """Simulate review process for demonstration purposes."""
        # In production, this would wait for actual human review
        # For now, return a pending status
        
        return {
            "review_status": "pending",
            "review_notes": f"Review {review_id} has been submitted and is awaiting human reviewer assignment.",
            "reviewer": "pending_assignment",
            "review_metadata": {
                "review_id": review_id,
                "status": "submitted",
                "estimated_completion": "24-48 hours",
                "submission_timestamp": datetime.now().isoformat()
            }
        }
    
    async def _save_review_record(self, sample_id: str, review_record: Dict[str, Any]) -> None:
        """Save review record to storage."""
        try:
            record_file = self.review_dir / f"review_record_{sample_id}.json"
            with open(record_file, 'w') as f:
                json.dump(review_record, f, indent=2)
            logger.info(f"Review record saved: {record_file}")
        except Exception as e:
            logger.error(f"Error saving review record: {e}")
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response for review failures."""
        return {
            "review_status": "error",
            "review_notes": f"Human review system error: {error_message}",
            "reviewer": "system_error",
            "review_metadata": {
                "error": True,
                "error_message": error_message,
                "error_timestamp": datetime.now().isoformat()
            }
        }


class ReviewInterface:
    """
    Utility class for managing review interfaces and workflows.
    """
    
    @staticmethod
    def load_pending_reviews(review_dir: Path) -> List[Dict[str, Any]]:
        """Load all pending review requests."""
        pending_reviews = []
        for review_file in review_dir.glob("review_*.json"):
            try:
                with open(review_file, 'r') as f:
                    review_data = json.load(f)
                    if review_data.get("review_metadata", {}).get("requires_review", False):
                        pending_reviews.append(review_data)
            except Exception as e:
                logger.error(f"Error loading review file {review_file}: {e}")
        
        return pending_reviews
    
    @staticmethod
    def submit_review_response(review_id: str, response: Dict[str, Any], review_dir: Path) -> bool:
        """Submit completed review response."""
        try:
            response_file = review_dir / f"response_{review_id}.json"
            response_data = {
                "review_id": review_id,
                "completion_timestamp": datetime.now().isoformat(),
                "response": response
            }
            
            with open(response_file, 'w') as f:
                json.dump(response_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error submitting review response: {e}")
            return False
