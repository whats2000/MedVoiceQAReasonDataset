"""
LangGraph-based pipeline for MedVoiceQA dataset processing.

Implements the complete workflow: Loader → Segmentation → ASR/TTS → Explanation → Validation → Human Review
"""

import logging
from typing import Dict, Any, Optional, List

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from nodes.asr_tts import run_asr_tts
from nodes.explanation import run_explanation
from nodes.human_review import run_human_review
from nodes.loader import run_loader
from nodes.segmentation import run_segmentation
from nodes.validation import run_validation

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """
    State schema for the MedVoiceQA pipeline.
    
    Each node adds its outputs to this state, following the contracts
    defined in the registry.json and README.
    """
    # Input fields
    sample_id: str

    # Loader outputs
    image_path: Optional[str]
    text_query: str
    metadata: Dict[str, Any]

    # Segmentation outputs
    visual_box: Optional[Dict[str, Any]]

    # ASR/TTS outputs
    speech_path: Optional[str]
    asr_text: Optional[str]
    speech_quality_score: Optional[float]

    # Explanation outputs
    text_explanation: Optional[str]
    uncertainty: Optional[float]

    # Validation outputs
    needs_review: Optional[bool]
    critic_notes: Optional[str]
    quality_scores: Optional[Dict[str, float]]

    # Human review outputs
    review_status: Optional[str]
    review_notes: Optional[str]
    approved: Optional[bool]

    # Node execution tracking
    completed_nodes: List[str]
    node_errors: Dict[str, str]

    # Additional context
    ground_truth_answer: Optional[str]
    processing_start_time: Optional[str]


def loader_node(state: PipelineState) -> Dict[str, Any]:
    """
    Load VQA-RAD sample and convert DICOM to PNG if needed.
    
    Consumes: sample_id
    Produces: image_path, text_query, metadata
    """
    logger.info(f"Running loader for sample: {state['sample_id']}")

    try:
        result = run_loader(
            sample_id=state["sample_id"],
            image_path=state.get("image_path"),  # May be pre-specified
            text_query=state.get("text_query"),  # May be pre-specified
            metadata=state.get("metadata", {})
        )

        # Update completed nodes
        completed_nodes = state.get("completed_nodes", [])
        completed_nodes.append("loader")

        return {
            "image_path": result["image_path"],
            "text_query": result["text_query"],
            "metadata": result["metadata"],
            "completed_nodes": completed_nodes,
            "node_name": "loader",
            "node_version": "v1.0.0",
        }

    except Exception as e:
        logger.error(f"Loader node failed: {e}")

        node_errors = state.get("node_errors", {})
        node_errors["loader"] = str(e)

        return {
            "node_errors": node_errors,
        }


def segmentation_node(state: PipelineState) -> Dict[str, Any]:
    """
    Perform visual localization using Gemini Vision.
    
    Consumes: image_path, text_query
    Produces: visual_box
    """
    logger.info(f"Running segmentation for sample: {state['sample_id']}")

    try:
        if not state.get("image_path"):
            logger.warning("No image available for segmentation")
            return {"visual_box": None}

        result = run_segmentation(
            image_path=state["image_path"],
            text_query=state["text_query"]
        )

        # Update completed nodes
        completed_nodes = state.get("completed_nodes", [])
        completed_nodes.append("segmentation")

        return {
            "visual_box": result["visual_box"],
            "completed_nodes": completed_nodes,
            "node_name": "segmentation",
            "node_version": "v1.0.0",
        }

    except Exception as e:
        logger.error(f"Segmentation node failed: {e}")

        node_errors = state.get("node_errors", {})
        node_errors["segmentation"] = str(e)

        return {
            "visual_box": None,
            "node_errors": node_errors,
        }


def asr_tts_node(state: PipelineState) -> Dict[str, Any]:
    """
    Generate speech from text and perform ASR validation.
    
    Consumes: text_query
    Produces: speech_path, asr_text, speech_quality_score
    """
    logger.info(f"Running ASR/TTS for sample: {state['sample_id']}")

    try:
        result = run_asr_tts(
            text_query=state["text_query"]
        )

        # Update completed nodes
        completed_nodes = state.get("completed_nodes", [])
        completed_nodes.append("asr_tts")

        return {
            "speech_path": result["speech_path"],
            "asr_text": result["asr_text"],
            "speech_quality_score": result["speech_quality_score"],
            "completed_nodes": completed_nodes,
            "node_name": "asr_tts",
            "node_version": "v1.0.0",
        }

    except Exception as e:
        logger.error(f"ASR/TTS node failed: {e}")

        node_errors = state.get("node_errors", {})
        node_errors["asr_tts"] = str(e)

        return {
            "speech_path": None,
            "asr_text": None,
            "speech_quality_score": None,
            "node_errors": node_errors,
        }


def explanation_node(state: PipelineState) -> Dict[str, Any]:
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
        )

        # Update completed nodes
        completed_nodes = state.get("completed_nodes", [])
        completed_nodes.append("explanation")

        return {
            "text_explanation": result["text_explanation"],
            "uncertainty": result["uncertainty"],
            "completed_nodes": completed_nodes,
            "node_name": "explanation",
            "node_version": "v1.0.0",
        }

    except Exception as e:
        logger.error(f"Explanation node failed: {e}")

        node_errors = state.get("node_errors", {})
        node_errors["explanation"] = str(e)

        return {
            "text_explanation": None,
            "uncertainty": None,
            "node_errors": node_errors,
        }


def validation_node(state: PipelineState) -> Dict[str, Any]:
    """
    Validate quality and determine if human review is needed.
    
    Consumes: all prior outputs
    Produces: needs_review, critic_notes, quality_scores
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
                "text_explanation": state.get("text_explanation"),
                "uncertainty": state.get("uncertainty"),
                "ground_truth_answer": state.get("ground_truth_answer"),
                "node_errors": state.get("node_errors", {})
            }
        )

        # Update completed nodes
        completed_nodes = state.get("completed_nodes", [])
        completed_nodes.append("validation")

        return {
            "needs_review": result["needs_review"],
            "critic_notes": result["critic_notes"],
            "quality_scores": result["quality_scores"],
            "completed_nodes": completed_nodes,
            "node_name": "validation",
            "node_version": "v1.0.0",
        }

    except Exception as e:
        logger.error(f"Validation node failed: {e}")

        node_errors = state.get("node_errors", {})
        node_errors["validation"] = str(e)

        return {
            "needs_review": True,  # Default to requiring review on validation failure
            "critic_notes": f"Validation failed: {e}",
            "quality_scores": {},
            "node_errors": node_errors,
        }


def human_review_node(state: PipelineState) -> Dict[str, Any]:
    """
    Human-in-the-loop review for samples that need attention.
    
    Consumes: all sample data + critic_notes
    Produces: review_status, review_notes, approved
    """
    logger.info(f"Running human review for sample: {state['sample_id']}")

    try:
        result = run_human_review(
            {
                "sample_data": state.copy(),  # Pass entire state for review
                "critic_notes": state.get("critic_notes", "No critic notes provided"),
            }
        )

        # Update completed nodes
        completed_nodes = state.get("completed_nodes", [])
        completed_nodes.append("human_review")

        return {
            "review_status": result["review_status"],
            "review_notes": result["review_notes"],
            "approved": result["approved"],
            "completed_nodes": completed_nodes,
            "node_name": "human_review",
            "node_version": "v1.0.0",
        }

    except Exception as e:
        logger.error(f"Human review node failed: {e}")

        node_errors = state.get("node_errors", {})
        node_errors["human_review"] = str(e)

        return {
            "review_status": "error",
            "review_notes": f"Human review failed: {e}",
            "approved": False,
            "node_errors": node_errors,
        }


def should_require_review(state: PipelineState) -> str:
    """
    Conditional edge function to determine if human review is needed.
    
    Returns:
        "human_review" if review is needed, END otherwise
    """
    needs_review = state.get("needs_review", False)
    has_errors = bool(state.get("node_errors", {}))

    if needs_review or has_errors:
        logger.info(f"Sample {state['sample_id']} requires human review")
        return "human_review"
    else:
        logger.info(f"Sample {state['sample_id']} passed validation")
        return END


def create_medvoice_pipeline() -> CompiledStateGraph:
    """
    Create the complete MedVoiceQA processing pipeline using LangGraph.
    
    Returns:
        Compiled LangGraph StateGraph ready for execution
    """
    # Create state graph
    graph_builder = StateGraph(PipelineState)

    # Add nodes
    graph_builder.add_node("loader", loader_node)
    graph_builder.add_node("segmentation", segmentation_node)
    graph_builder.add_node("asr_tts", asr_tts_node)
    graph_builder.add_node("explanation", explanation_node)
    graph_builder.add_node("validation", validation_node)
    graph_builder.add_node("human_review", human_review_node)

    # Define edges following the pipeline flow
    graph_builder.add_edge(START, "loader")
    graph_builder.add_edge("loader", "segmentation")
    graph_builder.add_edge("loader", "asr_tts")  # Parallel processing
    graph_builder.add_edge("segmentation", "explanation")
    graph_builder.add_edge("asr_tts", "explanation")
    graph_builder.add_edge("explanation", "validation")

    # Conditional edge for human review
    graph_builder.add_conditional_edges(
        "validation",
        should_require_review,
        {
            "human_review": "human_review",
            END: END,
        }
    )

    # Human review always goes to END
    graph_builder.add_edge("human_review", END)

    # Compile the graph
    pipeline = graph_builder.compile()

    logger.info("MedVoiceQA pipeline created successfully")
    return pipeline


def get_pipeline_info() -> Dict[str, Any]:
    """
    Get information about the pipeline structure and node versions.
    
    Returns:
        Pipeline metadata
    """
    return {
        "name": "MedVoiceQA Reasoning Dataset Pipeline",
        "version": "1.0.0",
        "description": "Transform VQA-RAD into multi-modal, explainable medical QA data",
        "nodes": [
            {"name": "loader", "version": "v1.0.0", "description": "Load VQA-RAD samples with DICOM to PNG conversion"},
            {"name": "segmentation", "version": "v1.0.0", "description": "Visual localization using Gemini Vision"},
            {"name": "asr_tts", "version": "v1.0.0", "description": "Speech synthesis and recognition"},
            {"name": "explanation", "version": "v1.0.0", "description": "Generate reasoning using Gemini"},
            {"name": "validation", "version": "v1.0.0", "description": "Quality assessment and validation"},
            {"name": "human_review", "version": "v1.0.0", "description": "Human-in-the-loop review"},
        ],
        "workflow": [
            "loader → segmentation",
            "loader → asr_tts",
            "segmentation → explanation",
            "asr_tts → explanation",
            "explanation → validation",
            "validation → human_review (conditional)",
        ],
        "quality_metrics": {
            "visual_box_iou": "> 0.50",
            "text_explanation_bertscore": "> 0.85",
            "consistency_rate": "≥ 80%",
            "overall_pass_rate": "≥ 80%",
        }
    }


if __name__ == "__main__":
    # Test pipeline creation
    pipeline = create_medvoice_pipeline()
    info = get_pipeline_info()

    print("Pipeline created successfully!")
    print(f"Name: {info['name']}")
    print(f"Nodes: {len(info['nodes'])}")
    print("Workflow:", " → ".join([step.split(" → ")[0] for step in info['workflow'][:3]]))
