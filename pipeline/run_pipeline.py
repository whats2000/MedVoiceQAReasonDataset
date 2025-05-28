"""
MedVoiceQA Reasoning Dataset Pipeline Runner

The Main entry point for running the LangGraph-based pipeline that transforms
VQA-RAD samples into multi-modal, explainable medical QA data.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from data.huggingface_loader import create_huggingface_dataset, HuggingFaceVQARADLoader
from pipeline.graph import create_medvoice_pipeline

# Load environment variables
load_dotenv()

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )


def validate_environment() -> Dict[str, bool]:
    """Validate required environment variables and dependencies."""
    checks = {
        "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
        "HUGGINGFACE_API_KEY": bool(os.getenv("HUGGINGFACE_API_KEY")),
    }

    missing = [key for key, value in checks.items() if not value]
    if missing:
        console.print(f"[red]Missing environment variables: {', '.join(missing)}[/red]")
        console.print("[yellow]Please check your .env file (see env.sample for reference)[/yellow]")

    return checks


def create_run_directory() -> Path:
    """Create a unique run directory with timestamp and hash."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create hash based on configuration
    config_str = f"{timestamp}_{os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    run_dir = Path("runs") / f"{timestamp}-{config_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def save_manifest(run_dir: Path, config: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Save run manifest for reproducibility."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results, "environment": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": os.name,
            "working_directory": str(Path.cwd()),
        },
        "models": {
            "gemini": os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
            "whisper": os.getenv("WHISPER_MODEL", "large-v3"),
            "bark": os.getenv("BARK_MODEL", "suno/bark"),
        }
    }

    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    console.print(f"[green]Manifest saved to {manifest_file}[/green]")


def run_pipeline(
    limit: Optional[int] = None,
    input_data: Optional[str] = None,
    dry_run: bool = False,
    use_huggingface: bool = True,
    log_level: str = "INFO"
) -> Dict[str, Any]:
    """
    Run the complete MedVoiceQA pipeline.

    Args:
        limit: Maximum number of samples to process
        input_data: Path to input dataset (if None, creates test data or downloads from HF)
        dry_run: If True, only validate setup without processing
        use_huggingface: If True, use Hugging Face VQA-RAD dataset
        log_level: Logging level

    Returns:
        Pipeline execution results
    """
    setup_logging(log_level)

    console.print("[bold blue]MedVoiceQA Reasoning Dataset Pipeline[/bold blue]")
    console.print("=" * 50)

    # Validate environment
    env_checks = validate_environment()
    if not all(env_checks.values()):
        if not dry_run:
            raise ValueError("Missing required environment variables")

    # Create run directory
    run_dir = create_run_directory()
    console.print(f"[green]Run directory: {run_dir}[/green]")

    # Setup logging file handler
    log_file = run_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logging.getLogger().addHandler(file_handler)

    config = {
        "limit": limit,
        "input_data": input_data,
        "dry_run": dry_run,
        "use_huggingface": use_huggingface,
        "log_level": log_level,
        "run_directory": str(run_dir),
    }

    # If dry runs, just validate and exit
    if dry_run:
        console.print("[yellow]Dry run mode - validating setup only[/yellow]")
        results = {"status": "dry_run", "validation": env_checks}
        save_manifest(run_dir, config, results)
        return results

    try:
        # Prepare input data
        if input_data is None:
            if use_huggingface:
                console.print("[yellow]No input data specified, using existing Hugging Face VQA-RAD data...[/yellow]")
                # Use existing HF data structure
                hf_data_dir = Path("data/vqarad_hf")
                if not hf_data_dir.exists():
                    console.print("[yellow]HF data not found, creating new dataset...[/yellow]")
                    hf_samples = limit or 300
                    input_data = str(run_dir / "hf_data")
                    hf_result = create_huggingface_dataset(
                        output_dir=input_data,
                        n_samples=hf_samples,
                        balance_by_modality=True,
                        balance_by_question_type=True,
                        random_seed=42
                    )

                    if "error" in hf_result:
                        raise ValueError(f"Failed to create Hugging Face dataset: {hf_result['error']}")

                    console.print(
                        f"[green]Created Hugging Face dataset with {hf_result['sample_count']} samples[/green]")
                else:
                    # Use existing HF data
                    input_data = str(hf_data_dir)
                    console.print(f"[green]Using existing Hugging Face data from {input_data}[/green]")
            else:
                console.print("[red]Non-HuggingFace mode not supported without sample_data module[/red]")
                raise ValueError("Please use --use-huggingface flag or provide --input-data")

        # Initialize dataset_metadata
        dataset_metadata = {}

        # Load samples from HF loader directly if using existing data
        if input_data == str(Path("data/vqarad_hf")):
            # Use HF loader to get samples directly
            hf_loader = HuggingFaceVQARADLoader()
            console.print("[yellow]Loading samples from Hugging Face VQA-RAD dataset...[/yellow]")

            all_samples = []
            sample_limit = limit or 300  # Default to 300 samples if no limit specified

            for i in range(sample_limit):
                try:
                    sample = hf_loader.get_sample_by_index(i)
                    if sample:
                        all_samples.append(sample)
                    else:
                        break  # No more samples available
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load sample {i}: {e}[/yellow]")
                    continue

            samples = all_samples
            console.print(f"[green]Loaded {len(samples)} samples from Hugging Face dataset[/green]")

            # Create metadata for HF samples
            dataset_metadata = {
                "source": "huggingface_vqa_rad",
                "sample_count": len(samples),
                "samples": samples
            }
        else:
            # Load samples from created dataset
            metadata_file = Path(input_data) / "sample_metadata.json"
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

            with open(metadata_file, 'r') as f:
                dataset_metadata = json.load(f)

            samples = dataset_metadata["samples"]
            if limit:
                samples = samples[:limit]

        console.print(f"[green]Processing {len(samples)} samples[/green]")

        # Create and run the pipeline
        pipeline = create_medvoice_pipeline()

        processed_samples = []
        failed_samples = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            task = progress.add_task("Processing samples...", total=len(samples))

            for i, sample in enumerate(samples):
                try:
                    progress.update(task, description=f"Processing {sample['sample_id']}...")

                    # Determine image path - handle both old and new format
                    image_path = None
                    if sample.get("copied_image"):
                        image_path = str(Path(input_data) / sample["copied_image"])
                    elif sample.get("image_path"):
                        image_path = sample["image_path"]

                    # Run pipeline on sample
                    result = pipeline.invoke({
                        "sample_id": sample["sample_id"],
                        "run_dir": str(run_dir),
                        "image_path": image_path,
                        "text_query": sample["question"],
                        "ground_truth_answer": sample.get("answer", ""),
                        "metadata": sample.get("metadata", {}),
                    })

                    # Add to processed samples
                    processed_samples.append({
                        "sample_id": sample["sample_id"],
                        "input": sample,
                        "output": result,
                        "status": "success"
                    })

                    progress.advance(task)

                except Exception as e:
                    logger.error(f"Failed to process sample {sample['sample_id']}: {e}")
                    failed_samples.append({
                        "sample_id": sample["sample_id"],
                        "error": str(e),
                        "status": "failed"
                    })
                    progress.advance(task)

        # Save results
        results_file = run_dir / "results.json"
        results = {
            "status": "completed",
            "processed_count": len(processed_samples),
            "failed_count": len(failed_samples),
            "processed_samples": processed_samples,
            "failed_samples": failed_samples,
            "dataset_metadata": dataset_metadata,
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Calculate summary statistics
        success_rate = len(processed_samples) / len(samples) if samples else 0

        console.print(f"[green]Pipeline completed![/green]")
        console.print(f"  - Processed: {len(processed_samples)}/{len(samples)} samples")
        console.print(f"  - Success rate: {success_rate:.1%}")
        console.print(f"  - Results saved to: {results_file}")

        if failed_samples:
            console.print(f"[yellow]  - Failed samples: {len(failed_samples)}[/yellow]")
            for failed in failed_samples[:3]:  # Show first 3 failures
                console.print(f"    - {failed['sample_id']}: {failed['error']}")
            if len(failed_samples) > 3:
                console.print(f"    ... and {len(failed_samples) - 3} more")

        # Save manifest
        save_manifest(run_dir, config, results)

        return results

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        console.print(f"[red]Pipeline failed: {e}[/red]")

        error_results = {
            "status": "failed",
            "error": str(e),
            "processed_count": 0,
            "failed_count": 0,
        }

        save_manifest(run_dir, config, error_results)
        raise


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(description="MedVoiceQA Reasoning Dataset Pipeline")
    parser.add_argument("--limit", type=int, help="Maximum number of samples to process")
    parser.add_argument("--input", help="Path to input dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without processing")
    parser.add_argument("--use-huggingface", action="store_true", default=True, help="Use Hugging Face VQA-RAD dataset")
    parser.add_argument("--no-huggingface", dest="use_huggingface", action="store_false",
                        help="Use local test data instead of Hugging Face")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    try:
        results = run_pipeline(
            limit=args.limit,
            input_data=args.input,
            dry_run=args.dry_run,
            use_huggingface=args.use_huggingface,
            log_level=args.log_level
        )

        if results["status"] == "completed":
            console.print("[bold green]✓ Pipeline completed successfully![/bold green]")
        elif results["status"] == "dry_run":
            console.print("[bold yellow]✓ Dry run completed - setup validated![/bold yellow]")
        else:
            console.print("[bold red]✗ Pipeline failed![/bold red]")
            exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        exit(1)
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        exit(1)


if __name__ == "__main__":
    main()
