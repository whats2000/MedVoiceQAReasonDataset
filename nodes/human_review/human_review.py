"""
Human Review Node for MedVoiceQA Pipeline.

Provides human-in-the-loop review for samples that need attention.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class HumanReviewNode:
    """
    Human-in-the-loop review for samples that need attention.
    
    This component handles samples flagged by the validation node for human review.
    It provides interfaces for human reviewers to assess and approve/reject samples.
    """

    def __init__(self, review_directory: Optional[str] = None):
        """
        Initialize the human review node.
        
        Args:
            review_directory: Directory to store review files. Default to runs/review/
        """
        self.review_directory = Path(review_directory or "runs/review")
        self.review_directory.mkdir(parents=True, exist_ok=True)

        # Review status options
        self.review_statuses = {
            "pending": "Awaiting human review",
            "approved": "Approved by human reviewer",
            "rejected": "Rejected by human reviewer",
            "needs_revision": "Needs revision before approval"
        }

    def create_review_request(self, sample_data: Dict[str, Any], critic_notes: str) -> str:
        """
        Create a review request file for human reviewers.
        
        Args:
            sample_data: Complete pipeline data for the sample
            critic_notes: Notes from the validation critic
            
        Returns:
            Path to the created review request file
        """
        try:
            # Generate unique review ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            sample_id = sample_data.get("sample_id", "unknown")
            review_id = f"{sample_id}_{timestamp}"

            # Create review data structure
            review_data = {
                "review_id": review_id,
                "created_at": datetime.now().isoformat(),
                "status": "pending",
                "sample_data": sample_data,
                "critic_notes": critic_notes,
                "reviewer_notes": "",
                "reviewer_id": "",
                "reviewed_at": None,
                "approval_reason": "",
                "rejection_reason": ""
            }

            # Save review request
            review_file = self.review_directory / f"{review_id}.json"
            with open(review_file, 'w', encoding='utf-8') as f:
                json.dump(review_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Created review request: {review_file}")
            return str(review_file)

        except Exception as e:
            logger.error(f"Failed to create review request: {e}")
            raise

    @staticmethod
    def load_review_request(review_file: str) -> Dict[str, Any]:
        """Load a review request from the file."""
        try:
            with open(review_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load review request {review_file}: {e}")
            raise

    def update_review_status(
        self,
        review_file: str,
        status: str,
        reviewer_id: str,
        reviewer_notes: str = "",
        approval_reason: str = "",
        rejection_reason: str = ""
    ) -> Dict[str, Any]:
        """
        Update the status of a review request.
        
        Args:
            review_file: Path to the review request file
            status: New status (approved/rejected/needs_revision)
            reviewer_id: ID of the human reviewer
            reviewer_notes: Additional notes from reviewer
            approval_reason: Reason for approval (if approved)
            rejection_reason: Reason for rejection (if rejected)
            
        Returns:
            Updated review data
        """
        try:
            # Load existing review data
            review_data = self.load_review_request(review_file)

            # Validate status
            if status not in self.review_statuses:
                raise ValueError(f"Invalid status: {status}. Must be one of {list(self.review_statuses.keys())}")

            # Update review data
            review_data.update({
                "status": status,
                "reviewer_id": reviewer_id,
                "reviewer_notes": reviewer_notes,
                "reviewed_at": datetime.now().isoformat(),
                "approval_reason": approval_reason,
                "rejection_reason": rejection_reason
            })

            # Save updated review data
            with open(review_file, 'w', encoding='utf-8') as f:
                json.dump(review_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Updated review {review_file} to status: {status}")
            return review_data

        except Exception as e:
            logger.error(f"Failed to update review status: {e}")
            raise

    def get_pending_reviews(self) -> list:
        """Get the list of all pending review requests."""
        try:
            pending_reviews = []
            for review_file in self.review_directory.glob("*.json"):
                try:
                    review_data = self.load_review_request(str(review_file))
                    if review_data.get("status") == "pending":
                        pending_reviews.append({
                            "file": str(review_file),
                            "review_id": review_data.get("review_id"),
                            "created_at": review_data.get("created_at"),
                            "sample_id": review_data.get("sample_data", {}).get("sample_id"),
                            "critic_notes": review_data.get("critic_notes", "")[:100] + "..."
                        })
                except Exception as e:
                    logger.warning(f"Failed to load review file {review_file}: {e}")
                    continue

            return sorted(pending_reviews, key=lambda x: x["created_at"])

        except Exception as e:
            logger.error(f"Failed to get pending reviews: {e}")
            return []

    @staticmethod
    def simulate_review(sample_data: Dict[str, Any], critic_notes: str) -> Dict[str, Any]:
        """
        Simulate human review for automated testing.
        
        This method provides automated review simulation based on simple heuristics.
        In production, this would be replaced by actual human review interface.
        """
        try:
            # Simple heuristics for automatic review simulation
            quality_scores = sample_data.get("quality_scores", {})
            uncertainty = sample_data.get("uncertainty", 0.5)

            overall_quality = quality_scores.get("overall_quality", 0.5)

            # Determine approval based on quality thresholds
            if overall_quality >= 0.8 and uncertainty <= 0.3:
                status = "approved"
                approval_reason = "High quality output with low uncertainty"
                rejection_reason = ""
            elif overall_quality >= 0.6 and uncertainty <= 0.5:
                status = "approved"
                approval_reason = "Acceptable quality output"
                rejection_reason = ""
            else:
                status = "rejected"
                approval_reason = ""
                rejection_reason = f"Quality too low: {overall_quality:.2f}, uncertainty too high: {uncertainty:.2f}"

            return {
                "review_status": status,
                "review_notes": f"Automated review simulation. Critic notes: {critic_notes[:100]}...",
                "approved": status == "approved"
            }

        except Exception as e:
            logger.error(f"Failed to simulate review: {e}")
            return {
                "review_status": "rejected",
                "review_notes": f"Error in review simulation: {str(e)}",
                "approved": False
            }


def run_human_review(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for human review.
    
    Args:
        state: Pipeline state containing sample_data and critic_notes
        
    Returns:
        Updated state with review_status, review_notes, and approved flag
    """
    logger.info("Running human review node")

    try:
        # Extract required data
        sample_data = state.copy()  # Use entire state as sample data
        critic_notes = state.get("critic_notes", "No critic notes provided")

        # Initialize human review node
        review_node = HumanReviewNode()

        # For now, use simulation mode
        # In production, this would create a review request and wait for human input
        review_result = review_node.simulate_review(sample_data, critic_notes)

        logger.info(f"Human review completed: {review_result['review_status']}")

        return {
            **state,
            "review_status": review_result["review_status"],
            "review_notes": review_result["review_notes"],
            "approved": review_result["approved"]
        }

    except Exception as e:
        logger.error(f"Human review node failed: {e}")
        return {
            **state,
            "review_status": "error",
            "review_notes": f"Error in human review: {str(e)}",
            "approved": False
        }
