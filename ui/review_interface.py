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
from PIL import Image, ImageDraw, ImageFont
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

    def draw_bounding_box_on_image(self, image: Image.Image, visual_box: Dict[str, Any]) -> Image.Image:
        """
        Draw bounding box on the medical image.
        
        Args:
            image: PIL Image object
            visual_box: Dictionary containing bounding box coordinates
            
        Returns:
            PIL Image with bounding box drawn
        """
        if not visual_box or not visual_box.get("bounding_box"):
            return image

        # Create a copy to avoid modifying the original
        img_with_bbox = image.copy()
        draw = ImageDraw.Draw(img_with_bbox)

        # Get image dimensions
        img_width, img_height = img_with_bbox.size

        # Get bounding box coordinates (normalized 0-1)
        bbox = visual_box["bounding_box"]
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        width = bbox.get("width", 0)
        height = bbox.get("height", 0)

        # Convert to pixel coordinates
        x1 = int(x * img_width)
        y1 = int(y * img_height)
        x2 = int((x + width) * img_width)
        y2 = int((y + height) * img_height)

        # Draw rectangle with bright red color for visibility
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Add confidence score if available
        confidence = visual_box.get("confidence")
        if confidence is not None:
            try:
                # Try to use a default font, fallback to default if not available
                font = ImageFont.load_default()
            except:
                font = None

            confidence_text = f"Confidence: {confidence:.2f}"

            # Position text above the bounding box
            text_x = x1
            text_y = max(0, y1 - 20)

            # Draw text background rectangle for better visibility
            if font:
                text_bbox = font.getbbox(confidence_text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            else:
                # Fallback dimensions
                text_width = len(confidence_text) * 8
                text_height = 12

            draw.rectangle([text_x, text_y, text_x + text_width + 4, text_y + text_height + 4],
                           fill="red", outline="red")
            draw.text((text_x + 2, text_y + 2), confidence_text, fill="white", font=font)

        return img_with_bbox

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

        # Get visual box at the start so it's available throughout the method
        visual_box = output_data.get("visual_box")

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

                        # Add toggle for bounding box overlay
                        if visual_box and visual_box.get("bounding_box"):
                            show_bbox = st.checkbox(
                                "Show Bounding Box Overlay",
                                value=True,
                                key=f"bbox_toggle_{sample_idx}",
                                help="Toggle to show/hide the AI-identified region of interest"
                            )

                            if show_bbox:
                                image_with_bbox = self.draw_bounding_box_on_image(image, visual_box)
                                st.image(image_with_bbox, caption=f"Medical Image with Bounding Box - {sample_id}",
                                         use_container_width=True)
                            else:
                                st.image(image, caption=f"Medical Image - {sample_id}", use_container_width=True)
                        else:
                            st.image(image, caption=f"Medical Image - {sample_id}", use_container_width=True)

                        # Show bounding box details if available
                        if visual_box:
                            with st.expander("🎯 Visual Localization Details & Manual Adjustment", expanded=False):
                                bbox = visual_box.get("bounding_box", {})
                                st.write("**Original AI Bounding Box Coordinates (normalized):**")
                                col_orig1, col_orig2 = st.columns(2)
                                with col_orig1:
                                    st.write(f"• X: {bbox.get('x', 'N/A')}")
                                    st.write(f"• Y: {bbox.get('y', 'N/A')}")
                                with col_orig2:
                                    st.write(f"• Width: {bbox.get('width', 'N/A')}")
                                    st.write(f"• Height: {bbox.get('height', 'N/A')}")

                                confidence = visual_box.get("confidence")
                                if confidence is not None:
                                    st.metric("Confidence", f"{confidence:.2f}")

                                st.divider()

                                # Manual bounding box adjustment with side-by-side layout
                                st.write("**🔧 Manual Bounding Box Adjustment:**")
                                st.caption(
                                    "Adjust the bounding box coordinates if the AI localization is incorrect. All values should be between 0 and 1 (normalized coordinates).")

                                # Get existing manual adjustments or use original values
                                existing_bbox_edit = existing_review.get("manual_bbox", bbox)

                                # Handle reset functionality with unique widget keys
                                reset_counter_key = f"reset_counter_{sample_idx}"
                                if reset_counter_key not in st.session_state:
                                    st.session_state[reset_counter_key] = 0

                                reset_counter = st.session_state[reset_counter_key]

                                # Create side-by-side layout: controls on left, preview on right
                                config_col, preview_col = st.columns([1, 1])
                                with config_col:
                                    st.write("**⚙️ Coordinates:**")

                                    # Input controls with dynamic keys that change on reset
                                    manual_x = st.number_input(
                                        "X coordinate",
                                        min_value=0.0,
                                        max_value=1.0,
                                        value=float(existing_bbox_edit.get('x', bbox.get('x', 0.0))),
                                        step=0.01,
                                        key=f"manual_x_{sample_idx}_{reset_counter}",
                                        help="Left edge of bounding box (0 = left edge, 1 = right edge)"
                                    )
                                    manual_y = st.number_input(
                                        "Y coordinate",
                                        min_value=0.0,
                                        max_value=1.0,
                                        value=float(existing_bbox_edit.get('y', bbox.get('y', 0.0))),
                                        step=0.01,
                                        key=f"manual_y_{sample_idx}_{reset_counter}",
                                        help="Top edge of bounding box (0 = top edge, 1 = bottom edge)"
                                    )
                                    manual_width = st.number_input(
                                        "Width",
                                        min_value=0.01,
                                        max_value=1.0,
                                        value=float(existing_bbox_edit.get('width', bbox.get('width', 0.1))),
                                        step=0.01,
                                        key=f"manual_width_{sample_idx}_{reset_counter}",
                                        help="Width of bounding box"
                                    )
                                    manual_height = st.number_input(
                                        "Height",
                                        min_value=0.01,
                                        max_value=1.0,
                                        value=float(existing_bbox_edit.get('height', bbox.get('height', 0.1))),
                                        step=0.01,
                                        key=f"manual_height_{sample_idx}_{reset_counter}",
                                        help="Height of bounding box"
                                    )

                                    # Validation for bounding box
                                    if manual_x + manual_width > 1.0:
                                        st.warning("⚠️ Box extends beyond image width")
                                    if manual_y + manual_height > 1.0:
                                        st.warning("⚠️ Box extends beyond image height")

                                    # Check if manual coordinates differ from original
                                    original_bbox = bbox
                                    coords_changed = (
                                        manual_x != original_bbox.get('x', 0.0) or
                                        manual_y != original_bbox.get('y', 0.0) or
                                        manual_width != original_bbox.get('width', 0.1) or
                                        manual_height != original_bbox.get('height', 0.1)
                                    )

                                    if coords_changed:
                                        st.info("📝 Modified from original")

                                    # Show coordinate summary
                                    coord_summary = f"**Current:** X={manual_x:.3f}, Y={manual_y:.3f}, W={manual_width:.3f}, H={manual_height:.3f}"
                                    st.caption(coord_summary)

                                    # Option to reset to original coordinates
                                    if coords_changed:
                                        if st.button(f"🔄 Reset to Original",
                                                     key=f"reset_bbox_{sample_idx}_{reset_counter}"):
                                            # Increment the reset counter to force new widgets with original values
                                            st.session_state[reset_counter_key] += 1
                                            st.rerun()

                                with preview_col:
                                    st.write("**🔍 Live Preview:**")
                                    st.caption("Updates automatically as you adjust coordinates")

                                    # Always show the preview with current coordinates
                                    manual_bbox_data = {
                                        "bounding_box": {
                                            "x": manual_x,
                                            "y": manual_y,
                                            "width": manual_width,
                                            "height": manual_height
                                        },
                                        "confidence": confidence,
                                        "manual_adjustment": coords_changed
                                    }

                                    try:
                                        preview_image = self.draw_bounding_box_on_image(image, manual_bbox_data)

                                        # Use different caption based on whether coordinates changed
                                        if coords_changed:
                                            caption = "🔧 With Manual Adjustments"
                                        else:
                                            caption = "👁️ Current Coordinates"

                                        st.image(preview_image, caption=caption, use_container_width=True)

                                    except Exception as e:
                                        st.error(f"Error generating preview: {e}")

                                st.divider()

                                region_desc = visual_box.get("region_description", "")
                                if region_desc:
                                    st.write("**Region Description:**")
                                    st.write(region_desc)

                                reasoning = visual_box.get("relevance_reasoning", "")
                                if reasoning:
                                    st.write("**Relevance Reasoning:**")
                                    st.write(reasoning)

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
                    st.metric("Speech Quality",
                              f"{speech_quality:.2f}")

        # Explanation and reasoning - with editing capability
        st.write("**AI Explanation:**")
        explanation = output_data.get("text_explanation", "No explanation available")

        # Check if there's an existing manual edit
        existing_explanation_edit = existing_review.get("manual_explanation")

        # Create tabs for original vs edited explanation
        exp_tab1, exp_tab2 = st.tabs(["📝 Edit Explanation", "👁️ Original AI Explanation"])

        with exp_tab1:
            st.caption("Edit the AI-generated explanation to correct any medical inaccuracies or improve clarity:")
            edited_explanation = st.text_area(
                "Corrected Explanation",
                value=existing_explanation_edit if existing_explanation_edit else explanation,
                height=150,
                key=f"edit_explanation_{sample_idx}",
                help="Modify the explanation to ensure medical accuracy. Your edits will be saved with the review."
            )

            # Show character count and basic validation
            char_count = len(edited_explanation) if edited_explanation else 0
            if char_count < 20:
                st.warning("⚠️ Explanation seems too short. Consider adding more detail.")
            elif char_count > 1000:
                st.warning("⚠️ Explanation is very long. Consider making it more concise.")
            else:
                st.success(f"✅ Good length: {char_count} characters")

            # Check if explanation was modified
            if edited_explanation != explanation:
                st.info("📝 Explanation has been modified from the original AI version.")

        with exp_tab2:
            st.write("**Original AI Generated Explanation:**")
            st.text_area(
                "Original",
                value=explanation,
                height=100,
                disabled=True,
                key=f"original_explanation_{sample_idx}"
            )

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
            image_issues = st.checkbox("Image Quality Issues", value=issues.get("image_quality", False),
                                       key=f"img_issue_{sample_idx}")
            text_issues = st.checkbox("Text/Explanation Issues", value=issues.get("text_quality", False),
                                      key=f"text_issue_{sample_idx}")

        with issue_cols[1]:
            audio_issues = st.checkbox("Audio Quality Issues", value=issues.get("audio_quality", False),
                                       key=f"audio_issue_{sample_idx}")
            bbox_issues = st.checkbox("Bounding Box Issues", value=issues.get("bbox_accuracy", False),
                                      key=f"bbox_issue_{sample_idx}")

        with issue_cols[2]:
            reasoning_issues = st.checkbox("Reasoning Issues", value=issues.get("reasoning_quality", False),
                                           key=f"reason_issue_{sample_idx}")
            other_issues = st.checkbox("Other Issues", value=issues.get("other", False),
                                       key=f"other_issue_{sample_idx}")
        # Save review button
        if st.button(f"Save Review for {sample_id}", key=f"save_{sample_idx}"):
            # Get manual bounding box values
            manual_bbox = {
                "x": st.session_state.get(f"manual_x_{sample_idx}",
                                          visual_box.get("bounding_box", {}).get("x", 0.0)) if visual_box else 0.0,
                "y": st.session_state.get(f"manual_y_{sample_idx}",
                                          visual_box.get("bounding_box", {}).get("y", 0.0)) if visual_box else 0.0,
                "width": st.session_state.get(f"manual_width_{sample_idx}",
                                              visual_box.get("bounding_box", {}).get("width",
                                                                                     0.1)) if visual_box else 0.1,
                "height": st.session_state.get(f"manual_height_{sample_idx}",
                                               visual_box.get("bounding_box", {}).get("height",
                                                                                      0.1)) if visual_box else 0.1,
            }

            # Get edited explanation
            manual_explanation = st.session_state.get(f"edit_explanation_{sample_idx}", explanation)

            # Check if there were manual edits
            bbox_manually_edited = visual_box and manual_bbox != visual_box.get("bounding_box", {})
            explanation_manually_edited = manual_explanation != explanation

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
                "manual_bbox": manual_bbox if bbox_manually_edited else None,
                "manual_explanation": manual_explanation if explanation_manually_edited else None,
                "has_manual_edits": {
                    "bbox": bbox_manually_edited,
                    "explanation": explanation_manually_edited,
                },
                "reviewed_at": datetime.now().isoformat(),
                "reviewer": st.session_state.get("reviewer_name", "anonymous")
            }
            self.save_review_data()

            # Show success message with details about what was saved
            success_msg = f"Review saved for {sample_id}"
            if bbox_manually_edited or explanation_manually_edited:
                success_msg += "\n🔧 Manual edits saved:"
                if bbox_manually_edited:
                    success_msg += "\n  • Bounding box coordinates adjusted"
                if explanation_manually_edited:
                    success_msg += "\n  • Explanation text corrected"

            st.success(success_msg)
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
                                        index=available_runs.index(
                                            current_selection) if current_selection in available_runs else 0)

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
