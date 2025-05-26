"""
Human Verification UI for MedVoiceQA Dataset

A Streamlit-based interface for reviewing and validating processed samples
from the MedVoiceQA pipeline before final dataset publication.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class MedVoiceQAReviewer:
    """Main class for the human verification interface."""
    
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = Path(runs_dir)
        self.current_run = None
        self.samples_data = []
        self.review_data = {}
        
    def load_run_data(self, run_id: str) -> bool:
        """Load data from a specific pipeline run."""
        run_path = self.runs_dir / run_id
        
        if not run_path.exists():
            st.error(f"Run directory not found: {run_path}")
            return False
            
        # Load results
        results_file = run_path / "results.json"
        if not results_file.exists():
            st.error(f"Results file not found: {results_file}")
            return False
            
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
                
            self.current_run = run_id
            self.samples_data = results.get("processed_samples", [])
            
            # Load existing review data if available
            review_file = run_path / "human_review.json"
            if review_file.exists():
                with open(review_file, 'r') as f:
                    self.review_data = json.load(f)
            else:
                self.review_data = {}
                
            return True
            
        except Exception as e:
            st.error(f"Error loading run data: {e}")
            return False
    
    def save_review_data(self):
        """Save current review decisions to file."""
        if not self.current_run:
            return
            
        review_file = self.runs_dir / self.current_run / "human_review.json"
        
        try:
            with open(review_file, 'w') as f:
                json.dump(self.review_data, f, indent=2)
            st.success("Review data saved successfully!")
            
        except Exception as e:
            st.error(f"Error saving review data: {e}")
    
    def get_sample_stats(self) -> Dict[str, Any]:
        """Calculate statistics for the current dataset."""
        if not self.samples_data:
            return {}
            
        total_samples = len(self.samples_data)
        reviewed_samples = len(self.review_data)
        approved_samples = sum(1 for r in self.review_data.values() if r.get("approved", False))
        
        # Quality scores statistics
        quality_scores = []
        for sample in self.samples_data:
            output = sample.get("output", {})
            scores = output.get("quality_scores", {})
            if scores:
                quality_scores.append(scores)
        
        return {
            "total_samples": total_samples,
            "reviewed_samples": reviewed_samples,
            "approved_samples": approved_samples,
            "review_progress": reviewed_samples / total_samples if total_samples > 0 else 0,
            "approval_rate": approved_samples / reviewed_samples if reviewed_samples > 0 else 0,
            "quality_scores": quality_scores,
        }
    
    def render_sample_card(self, sample: Dict[str, Any], sample_idx: int):
        """Render a single sample for review."""
        sample_id = sample["sample_id"]
        input_data = sample["input"]
        output_data = sample["output"]
        
        # Get existing review if available
        existing_review = self.review_data.get(sample_id, {})
        
        st.subheader(f"Sample: {sample_id}")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        with col1:
            # Display image if available
            image_path = output_data.get("image_path")
            if image_path:
                # Convert to absolute path if relative
                if not Path(image_path).is_absolute():
                    image_path = Path.cwd() / image_path
                else:
                    image_path = Path(image_path)
                    
                if image_path.exists():
                    try:
                        image = Image.open(image_path)
                        st.image(image, caption=f"Medical Image - {sample_id}", use_column_width=True)
                        
                        # Show bounding box if available
                        visual_box = output_data.get("visual_box")
                        if visual_box:
                            st.json(visual_box, expanded=False)
                            
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                else:
                    st.error(f"Image not found: {image_path}")
            else:
                st.info("No image available for this sample")
            
            # Audio player for speech
            speech_path = output_data.get("speech_path")
            if speech_path:
                # Convert to absolute path if relative
                if not Path(speech_path).is_absolute():
                    speech_path = Path.cwd() / speech_path
                else:
                    speech_path = Path(speech_path)
                    
                if speech_path.exists():
                    try:
                        with open(speech_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/wav")
                    except Exception as e:
                        st.error(f"Error loading audio: {e}")
                else:
                    st.error(f"Audio not found: {speech_path}")
            else:
                st.info("No audio available for this sample")
        
        with col2:
            # Sample metadata
            st.write("**Question:**")
            st.write(input_data.get("question", "N/A"))
            
            st.write("**Ground Truth Answer:**")
            st.write(input_data.get("answer", "N/A"))
            
            # ASR Text
            asr_text = output_data.get("asr_text")
            if asr_text:
                st.write("**ASR Text:**")
                st.write(asr_text)
                
                # ASR quality score
                speech_quality = output_data.get("speech_quality_score")
                if speech_quality is not None:
                    st.metric("Speech Quality", f"{speech_quality:.2f}")
        
        # Explanation and reasoning
        st.write("**AI Explanation:**")
        explanation = output_data.get("text_explanation", "No explanation available")
        st.write(explanation)
        
        # Uncertainty score
        uncertainty = output_data.get("uncertainty")
        if uncertainty is not None:
            st.metric("Uncertainty Score", f"{uncertainty:.2f}")
        
        # Quality scores
        quality_scores = output_data.get("quality_scores", {})
        if quality_scores:
            st.write("**Quality Scores:**")
            cols = st.columns(len(quality_scores))
            for i, (metric, score) in enumerate(quality_scores.items()):
                with cols[i]:
                    st.metric(metric.replace("_", " ").title(), f"{score:.2f}")
        
        # Validation results
        needs_review = output_data.get("needs_review", True)
        critic_notes = output_data.get("critic_notes", "")
        
        if needs_review:
            st.warning("⚠️ This sample was flagged for review")
        else:
            st.success("✅ This sample passed automatic validation")
            
        if critic_notes:
            st.write("**Validation Notes:**")
            st.info(critic_notes)
        
        st.divider()
        
        # Human review section
        st.write("**Human Review**")
        
        review_col1, review_col2 = st.columns(2)
        
        with review_col1:
            # Approval decision
            approved = st.checkbox(
                "Approve this sample",
                value=existing_review.get("approved", False),
                key=f"approve_{sample_idx}"
            )
        
        with review_col2:
            # Quality rating
            quality_rating = st.slider(
                "Quality Rating (1-5)",
                min_value=1,
                max_value=5,
                value=existing_review.get("quality_rating", 3),
                key=f"quality_{sample_idx}"
            )
        
        # Review notes
        review_notes = st.text_area(
            "Review Notes",
            value=existing_review.get("notes", ""),
            placeholder="Add any comments about this sample...",
            key=f"notes_{sample_idx}"
        )
        
        # Issues checkboxes
        st.write("**Issue Categories (check all that apply):**")
        
        issue_cols = st.columns(3)
        
        with issue_cols[0]:
            issues = existing_review.get("issues", {})
            image_issues = st.checkbox("Image Quality Issues", value=issues.get("image_quality", False), key=f"img_issue_{sample_idx}")
            text_issues = st.checkbox("Text/Explanation Issues", value=issues.get("text_quality", False), key=f"text_issue_{sample_idx}")
            
        with issue_cols[1]:
            audio_issues = st.checkbox("Audio Quality Issues", value=issues.get("audio_quality", False), key=f"audio_issue_{sample_idx}")
            bbox_issues = st.checkbox("Bounding Box Issues", value=issues.get("bbox_accuracy", False), key=f"bbox_issue_{sample_idx}")
            
        with issue_cols[2]:
            reasoning_issues = st.checkbox("Reasoning Issues", value=issues.get("reasoning_quality", False), key=f"reason_issue_{sample_idx}")
            other_issues = st.checkbox("Other Issues", value=issues.get("other", False), key=f"other_issue_{sample_idx}")
        
        # Save review button
        if st.button(f"Save Review for {sample_id}", key=f"save_{sample_idx}"):
            self.review_data[sample_id] = {
                "approved": approved,
                "quality_rating": quality_rating,
                "notes": review_notes,
                "issues": {
                    "image_quality": image_issues,
                    "text_quality": text_issues,
                    "audio_quality": audio_issues,
                    "bbox_accuracy": bbox_issues,
                    "reasoning_quality": reasoning_issues,
                    "other": other_issues,
                },
                "reviewed_at": datetime.now().isoformat(),
                "reviewer": st.session_state.get("reviewer_name", "anonymous")
            }
            self.save_review_data()
            st.success(f"Review saved for {sample_id}")
            st.rerun()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="MedVoiceQA Human Verification",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 MedVoiceQA Human Verification Interface")
    st.markdown("Review and validate processed samples from the MedVoiceQA pipeline")
    
    # Initialize reviewer with session state persistence
    if "reviewer" not in st.session_state:
        st.session_state.reviewer = MedVoiceQAReviewer()
    
    reviewer = st.session_state.reviewer
    
    # Sidebar for run selection and controls
    with st.sidebar:
        st.header("Controls")
        
        # Reviewer identification
        reviewer_name = st.text_input("Reviewer Name", value=st.session_state.get("reviewer_name", ""))
        if reviewer_name:
            st.session_state["reviewer_name"] = reviewer_name
        
        # Run selection
        st.subheader("Select Pipeline Run")
        
        # List available runs
        available_runs = []
        if reviewer.runs_dir.exists():
            for run_dir in reviewer.runs_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "results.json").exists():
                    available_runs.append(run_dir.name)
        
        available_runs.sort(reverse=True)  # Most recent first
        
        if available_runs:
            # Show current loaded run if any
            current_selection = reviewer.current_run if reviewer.current_run else available_runs[0]
            selected_run = st.selectbox("Available Runs", available_runs, 
                                      index=available_runs.index(current_selection) if current_selection in available_runs else 0)
            
            if st.button("Load Run Data"):
                if reviewer.load_run_data(selected_run):
                    st.session_state.reviewer = reviewer  # Update session state
                    st.success(f"Loaded data from run: {selected_run}")
                    st.rerun()
        else:
            st.warning("No pipeline runs found in the runs directory")
            st.stop()
        
        # Stats and progress
        if reviewer.samples_data:
            st.subheader("Review Progress")
            stats = reviewer.get_sample_stats()
            
            st.metric("Total Samples", stats["total_samples"])
            st.metric("Reviewed", f"{stats['reviewed_samples']}/{stats['total_samples']}")
            st.metric("Approved", stats["approved_samples"])
            
            # Progress bar
            progress = stats["review_progress"]
            st.progress(progress)
            st.write(f"Progress: {progress:.1%}")
            
            if stats["reviewed_samples"] > 0:
                st.metric("Approval Rate", f"{stats['approval_rate']:.1%}")
        
        # Export options
        if reviewer.review_data:
            st.subheader("Export")
            
            if st.button("Export Review Data"):
                # Create export data
                export_data = {
                    "run_id": reviewer.current_run,
                    "export_timestamp": datetime.now().isoformat(),
                    "reviewer": reviewer_name,
                    "stats": reviewer.get_sample_stats(),
                    "reviews": reviewer.review_data
                }
                
                # Download link
                export_json = json.dumps(export_data, indent=2)
                st.download_button(
                    label="Download Review Data (JSON)",
                    data=export_json,
                    file_name=f"medvoiceqa_review_{reviewer.current_run}.json",
                    mime="application/json"
                )
      # Main content area
    if not reviewer.samples_data:
        st.info("Please select and load a pipeline run from the sidebar to begin reviewing.")
        if reviewer.current_run:
            st.warning(f"Current run: {reviewer.current_run} - but no samples loaded. Check the results.json file.")
        st.stop()
    
    # Debug information
    st.success(f"✅ Loaded {len(reviewer.samples_data)} samples from run: {reviewer.current_run}")
    
    # Filtering and sorting options
    st.subheader("Filter and Sort Options")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        show_only_flagged = st.checkbox("Show only flagged samples")
        show_only_unreviewed = st.checkbox("Show only unreviewed samples")
    
    with filter_col2:
        sort_by = st.selectbox("Sort by", ["Sample ID", "Uncertainty", "Quality Score", "Review Status"])
    
    with filter_col3:
        samples_per_page = st.selectbox("Samples per page", [5, 10, 20, 50], index=1)
    
    # Filter samples
    filtered_samples = []
    for sample in reviewer.samples_data:
        sample_id = sample["sample_id"]
        output_data = sample["output"]
        
        # Apply filters
        if show_only_flagged and not output_data.get("needs_review", False):
            continue
            
        if show_only_unreviewed and sample_id in reviewer.review_data:
            continue
            
        filtered_samples.append(sample)
    
    # Sort samples
    if sort_by == "Uncertainty":
        filtered_samples.sort(key=lambda x: x["output"].get("uncertainty", 0), reverse=True)
    elif sort_by == "Quality Score":
        filtered_samples.sort(key=lambda x: sum(x["output"].get("quality_scores", {}).values()), reverse=True)
    elif sort_by == "Review Status":
        filtered_samples.sort(key=lambda x: x["sample_id"] in reviewer.review_data)
    
    # Pagination
    total_filtered = len(filtered_samples)
    total_pages = (total_filtered + samples_per_page - 1) // samples_per_page
    
    if total_pages > 1:
        page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1)) - 1
    else:
        page = 0
    
    start_idx = page * samples_per_page
    end_idx = min(start_idx + samples_per_page, total_filtered)
    
    page_samples = filtered_samples[start_idx:end_idx]
    
    st.write(f"Showing samples {start_idx + 1}-{end_idx} of {total_filtered}")
    
    # Render samples
    for i, sample in enumerate(page_samples):
        with st.container():
            reviewer.render_sample_card(sample, start_idx + i)
            st.markdown("---")
    
    # Summary statistics at the bottom
    if reviewer.review_data:
        st.subheader("Review Summary")
        
        stats = reviewer.get_sample_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviewed", stats["reviewed_samples"])
        with col2:
            st.metric("Approved", stats["approved_samples"])
        with col3:
            st.metric("Review Progress", f"{stats['review_progress']:.1%}")
        with col4:
            st.metric("Approval Rate", f"{stats['approval_rate']:.1%}")
        
        # Issues breakdown
        if stats["reviewed_samples"] > 0:
            st.subheader("Issue Categories")
            
            issue_counts = {}
            for review in reviewer.review_data.values():
                issues = review.get("issues", {})
                for issue_type, has_issue in issues.items():
                    if has_issue:
                        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            if issue_counts:
                # Create bar chart of issues
                issue_df = pd.DataFrame(list(issue_counts.items()), columns=["Issue Type", "Count"])
                fig = px.bar(issue_df, x="Issue Type", y="Count", title="Common Issues Found")
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
