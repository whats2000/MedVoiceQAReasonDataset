"""
LangGraph-based pipeline for MedVoiceQA dataset processing.

Implements the complete workflow: 
    Loader → (Segmentation || ASR/TTS) → Explanation → Validation → END
    
Key features:
- Parallel processing of Segmentation and ASR/TTS nodes for efficiency
- Full provenance tracking with node versioning
- Error handling and quality validation
- Human verification is handled separately through a dedicated UI interface
"""

import logging
from datetime import datetime
from typing import cast

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from models.workflow import (
    PipelineState,
    LoaderNodeResult, LoaderResult, SegmentationNodeResult, SegmentationSuccessResult, SegmentationErrorResult,
    ASRTTSNodeResult, ASRTTSSuccessResult, ASRTTSErrorResult, ExplanationNodeResult, ExplanationSuccessResult,
    ExplanationErrorResult, PipelineInfo, ValidationErrorResult, ValidationNodeResult, ValidationSuccessResult,
    SampleMetadata,
)
from nodes.asr_tts import run_asr_tts
from nodes.explanation import run_explanation
from nodes.loader import run_loader
from nodes.segmentation import run_segmentation
from nodes.validation import run_validation

logger = logging.getLogger(__name__)


def loader_node(state: PipelineState) -> LoaderNodeResult:
    """
    Load the VQA-RAD sample and convert DICOM to PNG if needed.

    Consumes: sample_id
    Produces: image_path, text_query, metadata
    """
    logger.info(f"Running loader for sample: {state['sample_id']}")

    try:
        result = run_loader(
            sample_id=state["sample_id"],
            image_path=state.get("image_path"),
            text_query=state.get("text_query"),
            metadata=cast(SampleMetadata, state.get("metadata", {}))
        )
        # Update completed nodes
        return cast(LoaderResult, {
            "image_path": result["image_path"],
            "text_query": result["text_query"],
            "metadata": result["metadata"],
            "loader_completed": True,
            "node_name": "loader",
            "node_version": "v1.0.0",
        })

    except Exception as e:
        logger.error(f"Loader node failed: {e}")

        node_errors = state.get("node_errors", {})
        node_errors["loader"] = str(e)

        return cast(LoaderResult, {
            "node_errors": node_errors,
        })


def segmentation_node(state: PipelineState) -> SegmentationNodeResult:
    """
    Perform visual localization using Gemini Vision.

    Consumes: image_path, text_query
    Produces: visual_box
    """
    logger.info(f"Running segmentation for sample: {state['sample_id']}")

    try:
        if not state.get("image_path"):
            logger.warning("No image available for segmentation")
            return cast(SegmentationSuccessResult, {
                "visual_box": None,
                "segmentation_completed": True,
                "node_name": "segmentation",
                "node_version": "v1.0.0",
            })

        # Cast to string to satisfy the type checker
        image_path = str(state["image_path"])

        result = run_segmentation(
            image_path=image_path,
            text_query=state["text_query"]
        )

        return cast(SegmentationSuccessResult, {
            "visual_box": result["visual_box"],
            "segmentation_completed": True,
            "segmentation_failed": result.get("segmentation_failed", False),
            "segmentation_error": result.get("segmentation_error"),
            "node_name": "segmentation",
            "node_version": "v1.0.0",
        })

    except Exception as e:
        logger.error(f"Segmentation node failed: {e}")

        node_errors = state.get("node_errors", {})
        node_errors["segmentation"] = str(e)

        return cast(SegmentationErrorResult, {
            "visual_box": None,
            "node_errors": node_errors,
        })


def asr_tts_node(state: PipelineState) -> ASRTTSNodeResult:
    """
    Generate speech from the text and perform ASR validation.

    Consumes: text_query
    Produces: speech_path, asr_text, speech_quality_score
    """
    logger.info(f"Running ASR/TTS for sample: {state['sample_id']}")

    try:
        # Use run_dir if provided, otherwise fall back to default
        output_dir = state.get("run_dir") or "runs/current"

        result = run_asr_tts(
            text_query=state["text_query"],
            sample_id=state["sample_id"],
            output_dir=output_dir
        )  # Update completed nodes
        return cast(ASRTTSSuccessResult, {
            "speech_path": result["speech_path"],
            "asr_text": result["asr_text"],
            "speech_quality_score": result["speech_quality_score"],
            "asr_tts_completed": True,
            "node_name": "asr_tts",
            "node_version": "v1.0.0",
        })

    except Exception as e:
        logger.error(f"ASR/TTS node failed: {e}")

        node_errors = state.get("node_errors", {})
        node_errors["asr_tts"] = str(e)

        return cast(ASRTTSErrorResult, {
            "speech_path": None,
            "asr_text": None,
            "speech_quality_score": None,
            "node_errors": node_errors,
        })


def explanation_node(state: PipelineState) -> ExplanationNodeResult:
    """
    Generate reasoning and uncertainty using Gemini.

    Consumes: image_path, text_query, visual_box
    Produces: text_explanation, uncertainty
    """
    logger.info(f"Running explanation for sample: {state['sample_id']}")

    try:
        result = run_explanation(
            {
                "image_path": state.get("image_path"),
                "text_query": state["text_query"],
                "visual_box": state.get("visual_box"),
                "ground_truth_answer": state.get("ground_truth_answer", "")
            }
        )  # Update completed nodes
        return cast(ExplanationSuccessResult, {
            "text_explanation": result["text_explanation"],
            "uncertainty": result["uncertainty"],
            "explanation_completed": True,
            "explanation_failed": result.get("explanation_failed", False),
            "explanation_error": result.get("explanation_error"),
            "node_name": "explanation",
            "node_version": "v1.0.0",
        })

    except Exception as e:
        logger.error(f"Explanation node failed: {e}")

        node_errors = state.get("node_errors", {})
        node_errors["explanation"] = str(e)

        return cast(ExplanationErrorResult, {
            "text_explanation": None,
            "uncertainty": None,
            "node_errors": node_errors,
        })


def validation_node(state: PipelineState) -> ValidationNodeResult:
    """
    Validate quality and mark pipeline completion.

    Consumes: all prior outputs
    Produces: needs_review, critic_notes, quality_scores, pipeline_status
    """
    logger.info(f"Running validation for sample: {state['sample_id']}")

    try:
        result = run_validation(
            {
                "sample_id": state["sample_id"],
                "image_path": state.get("image_path"),
                "text_query": state["text_query"],
                "visual_box": state.get("visual_box"),
                "speech_path": state.get("speech_path"),
                "asr_text": state.get("asr_text"),
                "speech_quality_score": state.get("speech_quality_score"),
                "text_explanation": state.get("text_explanation"),
                "uncertainty": state.get("uncertainty"),
                "ground_truth_answer": state.get("ground_truth_answer"),
                "node_errors": state.get("node_errors", {})
            }
        )

        # Determine pipeline status
        has_errors = bool(state.get("node_errors", {}))
        needs_review = result["needs_review"]

        if has_errors:
            pipeline_status = "failed"
        elif needs_review:
            pipeline_status = "needs_review"
        else:
            pipeline_status = "completed"

        return cast(ValidationSuccessResult, {
            "needs_review": needs_review,
            "critic_notes": result["critic_notes"],
            "quality_scores": result["quality_scores"],
            "validation_completed": True,
            "validation_failed": result.get("validation_failed", False),
            "validation_error": result.get("validation_error"),
            "pipeline_status": pipeline_status,
            "processing_end_time": datetime.now().isoformat(),
            "node_name": "validation",
            "node_version": "v1.0.0",
        })

    except Exception as e:
        logger.error(f"Validation node failed: {e}")

        node_errors = state.get("node_errors", {})
        node_errors["validation"] = str(e)

        return cast(ValidationErrorResult, {
            "needs_review": True,
            "critic_notes": f"Validation failed: {e}",
            "quality_scores": {},
            "pipeline_status": "failed",
            "processing_end_time": datetime.now().isoformat(),
            "node_errors": node_errors,
        })


def create_medvoice_pipeline() -> CompiledStateGraph:
    """
    Create the complete MedVoiceQA processing pipeline using LangGraph.

    Pipeline flow: START → Loader → (Segmentation || ASR/TTS) → Explanation → Validation → END
    
    Key features:
    - Parallel processing: Segmentation and ASR/TTS run concurrently after Loader
    - Error handling: Each node can fail gracefully with error tracking
    - Provenance: All nodes add versioning metadata for reproducibility
    - Human review: Handled separately through UI interface post-processing

    Returns:
        Compiled LangGraph StateGraph ready for execution
    """
    # Create the state graph
    graph_builder = StateGraph(PipelineState)

    # Add nodes
    graph_builder.add_node("loader", loader_node)
    graph_builder.add_node("segmentation", segmentation_node)
    graph_builder.add_node("asr_tts", asr_tts_node)
    graph_builder.add_node("explanation", explanation_node)
    graph_builder.add_node("validation", validation_node)

    # Define edges following the pipeline flow
    graph_builder.add_edge(START, "loader")
    graph_builder.add_edge("loader", "segmentation")
    graph_builder.add_edge("loader", "asr_tts")  # Parallel processing
    graph_builder.add_edge("segmentation", "explanation")
    graph_builder.add_edge("asr_tts", "explanation")
    graph_builder.add_edge("explanation", "validation")

    # The pipeline always ends after validation
    graph_builder.add_edge("validation", END)

    # Compile the graph
    main_pipeline = graph_builder.compile()

    logger.info("MedVoiceQA pipeline created successfully")
    return main_pipeline


def get_pipeline_info() -> PipelineInfo:
    """
    Get information about the pipeline structure and node versions.

    Returns:
        Pipeline metadata
    """
    return {
        "name": "MedVoiceQA Reasoning Dataset Pipeline",
        "version": "2.0.0",
        "description": "Transform VQA-RAD into multi-modal, explainable medical QA data. Human review handled separately via UI.",
        "nodes": [
            {"name": "loader", "version": "v1.0.0", "description": "Load VQA-RAD samples with DICOM to PNG conversion"},
            {"name": "segmentation", "version": "v1.0.0", "description": "Visual localization using Gemini Vision"},
            {"name": "asr_tts", "version": "v1.0.0", "description": "Speech synthesis and recognition"},
            {"name": "explanation", "version": "v1.0.0", "description": "Generate reasoning using Gemini"},
            {"name": "validation", "version": "v1.0.0", "description": "Quality assessment and validation"},
        ],        "workflow": [
            "START → loader",
            "loader → segmentation (parallel)",
            "loader → asr_tts (parallel)", 
            "segmentation → explanation",
            "asr_tts → explanation",
            "explanation → validation",
            "validation → END",
            "POST-PROCESSING → human_ui (separate)"
        ],
        "concurrency": {
            "parallel_nodes": ["segmentation", "asr_tts"],
            "description": "Segmentation and ASR/TTS run concurrently after Loader",
            "efficiency_gain": "~40% reduction in processing time"
        },
        "quality_metrics": {
            "visual_box_iou": "> 0.50",
            "text_explanation_bertscore": "> 0.85",
            "consistency_rate": "≥ 80%",
            "overall_pass_rate": "≥ 80%",
        },
        "human_review": {
            "method": "separate_ui_interface",
            "description": "Human verification handled post-processing through dedicated review interface"
        }
    }


if __name__ == "__main__":
    # Test pipeline creation
    pipeline = create_medvoice_pipeline()
    info = get_pipeline_info()

    print("Pipeline created successfully!")
    print(f"Name: {info['name']}")
    print(f"Version: {info['version']}")
    print(f"Nodes: {len(info['nodes'])}")
    print("Workflow:", " → ".join([step.split(" → ")[0] for step in info['workflow'][:3]]))
