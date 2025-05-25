"""
MedVoiceQA Reasoning Dataset Pipeline

This module contains the main LangGraph pipeline for processing VQA-RAD samples
into multi-modal, explainable medical QA data.
"""

import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Annotated

import typer
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from nodes.loader import VQARADLoader
from nodes.segmentation import GeminiVisionSegmenter
from nodes.asr_tts import BarkWhisperProcessor
from nodes.explanation import GeminiReasoningEngine
from nodes.validation import GeminiValidationDuo
from nodes.human_review import HumanReviewNode

# Load environment variables
load_dotenv()

console = Console()


class PipelineState(TypedDict):
    """State object for the MedVoiceQA pipeline"""
    
    # Input
    sample_id: str
    
    # Loader outputs
    image_path: Optional[str]
    text_query: Optional[str]
    metadata: Optional[Dict]
    
    # Segmentation outputs
    visual_box: Optional[Dict]
    
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
    quality_scores: Optional[Dict]
    
    # Human review outputs
    review_status: Optional[str]
    review_notes: Optional[str]
    approved: Optional[bool]
    
    # Pipeline metadata
    node_versions: Dict[str, str]
    processing_time: Dict[str, float]
    errors: List[str]


class MedVoiceQAPipeline:
    """Main pipeline orchestrator using LangGraph"""
    
    def __init__(self, config_path: str = "registry.json"):
        """Initialize pipeline with configuration"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.graph = self._build_graph()
        
        # Initialize nodes
        self.loader = VQARADLoader()
        self.segmenter = GeminiVisionSegmenter()
        self.asr_tts_processor = BarkWhisperProcessor()
        self.reasoning_engine = GeminiReasoningEngine()
        self.validator = GeminiValidationDuo()
        self.human_reviewer = HumanReviewNode()
        
        console.print("[bold green]✓[/bold green] Pipeline initialized successfully")
    
    def _load_config(self) -> Dict:
        """Load pipeline configuration from registry.json"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            return json.load(f)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        
        # Define the graph
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("loader", self._loader_node)
        workflow.add_node("segmentation", self._segmentation_node)
        workflow.add_node("asr_tts", self._asr_tts_node)
        workflow.add_node("explanation", self._explanation_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("human_review", self._human_review_node)
        
        # Set entry point
        workflow.set_entry_point("loader")
        
        # Add edges
        workflow.add_edge("loader", "segmentation")
        workflow.add_edge("loader", "asr_tts")
        workflow.add_edge("segmentation", "explanation")
        workflow.add_edge("asr_tts", "explanation")
        workflow.add_edge("explanation", "validation")
        
        # Conditional edge for human review
        workflow.add_conditional_edges(
            "validation",
            self._should_human_review,
            {
                "review": "human_review",
                "end": END
            }
        )
        
        workflow.add_edge("human_review", END)
        
        # Add memory for persistence
        memory = MemorySaver()
        
        return workflow.compile(checkpointer=memory)
    
    async def _loader_node(self, state: PipelineState) -> PipelineState:
        """Load VQA-RAD sample data"""
        start_time = datetime.now()
        
        try:
            result = await self.loader.process(state["sample_id"])
            
            state.update({
                "image_path": result["image_path"],
                "text_query": result["text_query"],
                "metadata": result["metadata"],
                "node_versions": {**state.get("node_versions", {}), "loader": "v1.0.0"},
                "processing_time": {
                    **state.get("processing_time", {}),
                    "loader": (datetime.now() - start_time).total_seconds()
                }
            })
            
        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Loader error: {str(e)}"]
            console.print(f"[bold red]✗[/bold red] Loader failed: {e}")
        
        return state
    
    async def _segmentation_node(self, state: PipelineState) -> PipelineState:
        """Visual localization using Gemini Vision"""
        start_time = datetime.now()
        
        try:
            result = await self.segmenter.process(
                image_path=state["image_path"],
                text_query=state["text_query"]
            )
            
            state.update({
                "visual_box": result["visual_box"],
                "node_versions": {**state.get("node_versions", {}), "segmentation": "v1.0.0"},
                "processing_time": {
                    **state.get("processing_time", {}),
                    "segmentation": (datetime.now() - start_time).total_seconds()
                }
            })
            
        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Segmentation error: {str(e)}"]
            console.print(f"[bold red]✗[/bold red] Segmentation failed: {e}")
        
        return state
    
    async def _asr_tts_node(self, state: PipelineState) -> PipelineState:
        """Speech synthesis and recognition"""
        start_time = datetime.now()
        
        try:
            result = await self.asr_tts_processor.process(
                text_query=state["text_query"]
            )
            
            state.update({
                "speech_path": result["speech_path"],
                "asr_text": result["asr_text"],
                "speech_quality_score": result["speech_quality_score"],
                "node_versions": {**state.get("node_versions", {}), "asr_tts": "v1.0.0"},
                "processing_time": {
                    **state.get("processing_time", {}),
                    "asr_tts": (datetime.now() - start_time).total_seconds()
                }
            })
            
        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"ASR/TTS error: {str(e)}"]
            console.print(f"[bold red]✗[/bold red] ASR/TTS failed: {e}")
        
        return state
    
    async def _explanation_node(self, state: PipelineState) -> PipelineState:
        """Generate reasoning and uncertainty"""
        start_time = datetime.now()
        
        try:
            result = await self.reasoning_engine.process(
                image_path=state["image_path"],
                text_query=state["text_query"],
                visual_box=state["visual_box"]
            )
            
            state.update({
                "text_explanation": result["text_explanation"],
                "uncertainty": result["uncertainty"],
                "node_versions": {**state.get("node_versions", {}), "explanation": "v1.0.0"},
                "processing_time": {
                    **state.get("processing_time", {}),
                    "explanation": (datetime.now() - start_time).total_seconds()
                }
            })
            
        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Explanation error: {str(e)}"]
            console.print(f"[bold red]✗[/bold red] Explanation failed: {e}")
        
        return state
    
    async def _validation_node(self, state: PipelineState) -> PipelineState:
        """Quality validation and critic assessment"""
        start_time = datetime.now()
        
        try:
            result = await self.validator.process(
                image_path=state["image_path"],
                text_query=state["text_query"],
                visual_box=state["visual_box"],
                speech_path=state["speech_path"],
                asr_text=state["asr_text"],
                text_explanation=state["text_explanation"],
                uncertainty=state["uncertainty"]
            )
            
            state.update({
                "needs_review": result["needs_review"],
                "critic_notes": result["critic_notes"],
                "quality_scores": result["quality_scores"],
                "node_versions": {**state.get("node_versions", {}), "validation": "v1.0.0"},
                "processing_time": {
                    **state.get("processing_time", {}),
                    "validation": (datetime.now() - start_time).total_seconds()
                }
            })
            
        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Validation error: {str(e)}"]
            console.print(f"[bold red]✗[/bold red] Validation failed: {e}")
        
        return state
    
    async def _human_review_node(self, state: PipelineState) -> PipelineState:
        """Human-in-the-loop review"""
        start_time = datetime.now()
        
        try:
            result = await self.human_reviewer.process(
                sample_data=state,
                critic_notes=state["critic_notes"]
            )
            
            state.update({
                "review_status": result["review_status"],
                "review_notes": result["review_notes"],
                "approved": result["approved"],
                "node_versions": {**state.get("node_versions", {}), "human_review": "v1.0.0"},
                "processing_time": {
                    **state.get("processing_time", {}),
                    "human_review": (datetime.now() - start_time).total_seconds()
                }
            })
            
        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Human review error: {str(e)}"]
            console.print(f"[bold red]✗[/bold red] Human review failed: {e}")
        
        return state
    
    def _should_human_review(self, state: PipelineState) -> str:
        """Decide whether sample needs human review"""
        if state.get("needs_review", False):
            return "review"
        return "end"
    
    async def process_sample(self, sample_id: str) -> Dict:
        """Process a single sample through the pipeline"""
        
        initial_state = PipelineState(
            sample_id=sample_id,
            image_path=None,
            text_query=None,
            metadata=None,
            visual_box=None,
            speech_path=None,
            asr_text=None,
            speech_quality_score=None,
            text_explanation=None,
            uncertainty=None,
            needs_review=None,
            critic_notes=None,
            quality_scores=None,
            review_status=None,
            review_notes=None,
            approved=None,
            node_versions={},
            processing_time={},
            errors=[]
        )
        
        config = RunnableConfig(
            configurable={"thread_id": f"sample_{sample_id}"}
        )
        
        result = await self.graph.ainvoke(initial_state, config=config)
        
        return result
    
    async def process_batch(self, sample_ids: List[str], max_workers: int = 4) -> List[Dict]:
        """Process multiple samples in parallel"""
        
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_with_semaphore(sample_id: str):
            async with semaphore:
                return await self.process_sample(sample_id)
        
        tasks = [process_with_semaphore(sample_id) for sample_id in sample_ids]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Processing samples...", total=len(sample_ids))
            
            results = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                progress.advance(task)
        
        return results


def create_run_directory() -> Path:
    """Create a timestamped run directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_hash = str(hash(timestamp))[-6:]  # Simple hash for uniqueness
    
    run_dir = Path("runs") / f"{timestamp}-{run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def save_results(results: List[Dict], run_dir: Path) -> None:
    """Save processing results with manifest"""
    
    # Save individual results
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    for result in results:
        sample_id = result["sample_id"]
        output_file = results_dir / f"{sample_id}.json"
        
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
    
    # Create manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(results),
        "successful_samples": len([r for r in results if not r.get("errors")]),
        "failed_samples": len([r for r in results if r.get("errors")]),
        "needs_review_count": len([r for r in results if r.get("needs_review")]),
        "config_snapshot": "registry.json",
        "results_directory": str(results_dir),
        "pipeline_version": "v1.0.0"
    }
    
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    console.print(f"[bold green]✓[/bold green] Results saved to: {run_dir}")


async def main(
    limit: int = typer.Option(300, help="Maximum number of samples to process"),
    max_workers: int = typer.Option(4, help="Maximum parallel workers"),
    dry_run: bool = typer.Option(False, help="Run in dry-run mode (no actual processing)")
):
    """Run the MedVoiceQA Reasoning Dataset pipeline"""
    
    console.print("[bold blue]MedVoiceQA Reasoning Dataset Pipeline[/bold blue]")
    console.print(f"Processing up to {limit} samples with {max_workers} workers")
    
    if dry_run:
        console.print("[yellow]Running in dry-run mode[/yellow]")
        return
    
    # Create run directory
    run_dir = create_run_directory()
    console.print(f"Run directory: {run_dir}")
    
    # Initialize pipeline
    pipeline = MedVoiceQAPipeline()
    
    # Get sample IDs (placeholder - implement actual sample loading)
    # This would typically load from VQA-RAD dataset
    sample_ids = [f"sample_{i:03d}" for i in range(1, min(limit + 1, 301))]
    
    try:
        # Process samples
        results = await pipeline.process_batch(sample_ids, max_workers)
        
        # Save results
        save_results(results, run_dir)
        
        # Print summary
        success_count = len([r for r in results if not r.get("errors")])
        review_count = len([r for r in results if r.get("needs_review")])
        
        console.print("\n[bold green]Pipeline Summary:[/bold green]")
        console.print(f"  Total processed: {len(results)}")
        console.print(f"  Successful: {success_count}")
        console.print(f"  Needs review: {review_count}")
        console.print(f"  Pass rate: {success_count/len(results)*100:.1f}%")
        
    except Exception as e:
        console.print(f"[bold red]Pipeline failed:[/bold red] {e}")
        raise


if __name__ == "__main__":
    typer.run(main)
