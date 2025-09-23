from typing import Optional, TypedDict, Union, Dict


class NodeInfo(TypedDict):
    """Type definition for node information structure."""
    name: str
    version: str
    description: str


class ConcurrencyInfo(TypedDict):
    """Type definition for concurrency information structure."""
    parallel_nodes: list[str]
    description: str
    efficiency_gain: str


class QualityMetrics(TypedDict):
    """Type definition for quality metrics structure."""
    visual_box_iou: str
    text_explanation_bertscore: str
    consistency_rate: str
    overall_pass_rate: str


class HumanReviewInfo(TypedDict):
    """Type definition for human review information structure."""
    method: str
    description: str


class PipelineInfo(TypedDict):
    """Type definition for pipeline information structure."""
    name: str
    version: str
    description: str
    nodes: list[NodeInfo]
    workflow: list[str]
    concurrency: ConcurrencyInfo
    quality_metrics: QualityMetrics
    human_review: HumanReviewInfo


class SampleMetadata(TypedDict, total=False):
    """Type definition for sample metadata structure based on actual loader output."""
    original_index: int
    image_name: str
    source: str
    dataset_split: str
    loaded_by: str
    loader_version: str
    sample_id: str


class LoaderResult(TypedDict):
    """Type definition for loader node return values."""
    image_path: str
    text_query: str
    metadata: SampleMetadata
    loader_completed: bool
    node_name: str
    node_version: str


class NodeErrorResult(TypedDict):
    """Type definition for node error return values."""
    node_errors: Dict[str, str]


class BoundingBoxCoordinates(TypedDict):
    """Type definition for bounding box coordinates."""
    x: float
    y: float
    width: float
    height: float


class VisualBox(TypedDict):
    """Type definition for visual localization data."""
    bounding_box: BoundingBoxCoordinates
    confidence: float
    region_description: str
    relevance_reasoning: str


class QualityScoresResult(TypedDict):
    """Type definition for quality scores structure."""
    visual_localization_quality: float
    speech_processing_quality: float
    reasoning_quality: float
    consistency_score: float
    overall_quality: float


class SegmentationSuccessResult(TypedDict, total=False):
    """Type definition for successful segmentation node return values."""
    visual_box: Optional[VisualBox]
    segmentation_completed: bool
    segmentation_failed: bool
    segmentation_error: str
    node_name: str
    node_version: str


class SegmentationErrorResult(TypedDict):
    """Type definition for segmentation node error return values."""
    visual_box: None
    node_errors: Dict[str, str]


SegmentationNodeResult = Union[SegmentationSuccessResult, SegmentationErrorResult]


class ASRTTSSuccessResult(TypedDict, total=False):
    """Type definition for successful ASR/TTS node return values."""
    speech_path: Optional[str]
    asr_text: str
    speech_quality_score: float
    asr_tts_completed: bool
    node_name: str
    node_version: str


class ASRTTSErrorResult(TypedDict):
    """Type definition for ASR/TTS node error return values."""
    speech_path: None
    asr_text: None
    speech_quality_score: None
    node_errors: Dict[str, str]


ASRTTSNodeResult = Union[ASRTTSSuccessResult, ASRTTSErrorResult]


class ExplanationSuccessResult(TypedDict, total=False):
    """Type definition for successful explanation node return values."""
    text_explanation: str
    uncertainty: float
    explanation_completed: bool
    explanation_failed: bool
    explanation_error: str
    node_name: str
    node_version: str


class ExplanationErrorResult(TypedDict):
    """Type definition for explanation node error return values."""
    text_explanation: None
    uncertainty: None
    node_errors: Dict[str, str]


ExplanationNodeResult = Union[ExplanationSuccessResult, ExplanationErrorResult]


class ValidationSuccessResult(TypedDict, total=False):
    """Type definition for successful validation node return values."""
    needs_review: bool
    critic_notes: str
    quality_scores: QualityScoresResult
    validation_completed: bool
    validation_failed: bool
    validation_error: str
    pipeline_status: str
    processing_end_time: str
    node_name: str
    node_version: str


class ValidationErrorResult(TypedDict):
    """Type definition for validation node error return values."""
    needs_review: bool
    critic_notes: str
    quality_scores: QualityScoresResult
    pipeline_status: str
    processing_end_time: str
    node_errors: Dict[str, str]


ValidationNodeResult = Union[ValidationSuccessResult, ValidationErrorResult]
LoaderNodeResult = Union[LoaderResult, NodeErrorResult]


class PipelineState(TypedDict):
    """
    State schema for the MedVoiceQA pipeline.

    Each node adds its outputs to this state, following the contracts
    defined in the registry.json and README.
    """
    # Input fields
    sample_id: str
    run_dir: Optional[str]  # Directory where pipeline outputs should be saved

    # Loader outputs
    image_path: Optional[str]
    text_query: str
    metadata: SampleMetadata

    # Segmentation outputs
    visual_box: Optional[VisualBox]
    segmentation_failed: Optional[bool]
    segmentation_error: Optional[str]

    # ASR/TTS outputs
    speech_path: Optional[str]
    asr_text: Optional[str]
    speech_quality_score: Optional[float]

    # Explanation outputs
    text_explanation: Optional[str]
    uncertainty: Optional[float]
    explanation_failed: Optional[bool]
    explanation_error: Optional[str]

    # Validation outputs
    needs_review: Optional[bool]
    critic_notes: Optional[str]
    quality_scores: Optional[QualityScoresResult]
    validation_failed: Optional[bool]
    validation_error: Optional[str]

    # Node execution tracking
    loader_completed: Optional[bool]
    segmentation_completed: Optional[bool]
    asr_tts_completed: Optional[bool]
    explanation_completed: Optional[bool]
    validation_completed: Optional[bool]
    node_errors: Dict[str, str]

    # Additional context
    ground_truth_answer: Optional[str]
    processing_start_time: Optional[str]
    processing_end_time: Optional[str]
    pipeline_status: Optional[str]  # "completed", "failed", "needs_review"