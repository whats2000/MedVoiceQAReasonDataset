# MVVQ-RAD: Medical Voice Vision Question-Reason Answer Dataset

Transform [VQA‑RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) into a multi‑modal, explainable medical‑QA mini‑corpus (speech ✚ bounding box ✚ reasoning)

---

## 📝 ToDo

- [x] Implement the annotation pipeline using LangGraph
- [x] Implement the human verification UI
- [ ] Cooperate with medical institutions to validate the dataset
- [ ] Publish the dataset on Hugging Face
- [ ] Publish the full detailed paper with human validation results to ArXiv

---

## ⭐️ What’s inside?

| Modality        | Fields                             | Source models/tools                       |
|-----------------|------------------------------------|-------------------------------------------|
| **Image**       | `image` (PNG)                      | VQA‑RAD DICOM → PNG via **dicom2png**     |
| **Speech**      | `speech_input` (WAV) · `asr_text`  | **Bark** (TTS) → **Na0s Whisper‑L** (ASR) |
| **Visual loc.** | `visual_box`                       | **Gemini 2 Flash** Vision (bbox‑only)     |
| **Reasoning**   | `text_explanation` · `uncertainty` | **Gemini 2 Flash** Language               |
| **QA flag**     | `needs_review` · `critic_notes`    | Gemini validation duo                     |

> **Size:** 300 samples covering CT/MRI/X‑ray, stratified by modality & question type. (Number may increase after discussion with medical institutions)

---

## 🗺️ Pipeline (LangGraph)

```mermaid
flowchart TD
    START([START]) --> Loader[Loader Node<br/>Load VQA-RAD sample<br/>DICOM → PNG conversion]
    
    Loader --> |"image_path<br/>text_query<br/>metadata"| Segmentation[Segmentation Node<br/>Visual localization<br/>Gemini Vision bbox detection]
    Loader --> |"text_query<br/>sample_id"| ASR_TTS[ASR/TTS Node<br/>Bark TTS synthesis<br/>Whisper ASR validation]
    
    Segmentation --> |"visual_box"| Explanation[Explanation Node<br/>Reasoning generation<br/>Uncertainty estimation<br/>Gemini Language]
    ASR_TTS --> |"speech_path<br/>asr_text<br/>quality_score"| Explanation
    
    Explanation --> |"text_explanation<br/>uncertainty"| Validation[Validation Node<br/>Quality assessment<br/>Error detection<br/>Review flagging]
    
    Validation --> |"needs_review<br/>critic_notes<br/>quality_scores"| Pipeline_END([PIPELINE END])
    
    Pipeline_END -.-> |"Post-processing"| Human_UI[Human Verification UI<br/>Streamlit interface<br/>Sample review & approval<br/>Quality control]
    
    Human_UI --> Dataset[Final Dataset<br/>Validated samples<br/>Ready for publication]
    
    %% Styling
    classDef nodeStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef startEnd fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    classDef humanProcess fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,stroke-dasharray: 5 5
    classDef dataOutput fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class START,Pipeline_END startEnd
    class Loader,Segmentation,ASR_TTS,Explanation,Validation nodeStyle
    class Human_UI humanProcess
    class Dataset dataOutput
```

### 📊 Processing Details

| Stage | Concurrency | Input | Output | Models/Tools |
|-------|-------------|-------|--------|--------------|
| **Loader** | Sequential | `sample_id` | `image_path`, `text_query`, `metadata` | DICOM2PNG converter |
| **Segmentation** | **Parallel** | `image_path`, `text_query` | `visual_box` | Gemini 2 Flash Vision |
| **ASR/TTS** | **Parallel** | `text_query`, `sample_id` | `speech_path`, `asr_text`, `quality_score` | Bark TTS + Whisper-L ASR |
| **Explanation** | Sequential | All prior outputs | `text_explanation`, `uncertainty` | Gemini 2 Flash Language |
| **Validation** | Sequential | All outputs + errors | `needs_review`, `critic_notes`, `quality_scores` | Custom validation logic |
| **Human Review** | Manual | Validated samples | Final dataset | Streamlit UI interface |

*✨ **Key Feature:** Segmentation and ASR/TTS nodes run in **parallel** after the Loader, reducing total processing time by ~40%.*

*🔄 Each node appends versioning metadata (`node_name`, `node_version`) for full provenance tracking.*

---

## 🚀 Quick Start

### 1 · Clone & install with uv

> [!NOTE]
> If you have not installed `uv`, please do so first:
> [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone https://github.com/whats2000/MedVoiceQAReasonDataset.git
cd MedVoiceQAReasonDataset

# Check CUDA version
nvidia-smi
# It should show something like this:
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
# |-----------------------------------------+------------------------+----------------------+

# Install with uv (Please pick the right one for your CUDA version)
uv sync --extra cpu
# Or if you using cuda 11.8
uv sync --extra cu118 
# Or if you using cuda 12.6
uv sync --extra cu126
# Or if you using cuda 12.8
uv sync --extra cu128
```

### 2 · Prepare secrets

Create an `.env` file with your Gemini & Hugging Face keys (see [.env.example](.env.example)):

### 3. Download VQA‑RAD index

```bash
uv run .\data\huggingface_loader.py
```

### 4 · Verify installation

```bash
uv run pytest
```

Outputs land in `runs/<timestamp>-<hash>/` with `manifest.json` for reproducibility.


### 5 · Dry‑run on 50 samples

```bash
uv run python pipeline/run_pipeline.py --limit 50
```

### 6 · Full 300‑sample run

```bash
uv run python pipeline/run_pipeline.py
```

### 7 · Human verification via UI

After processing, review the generated data through the web interface:

```bash
# Install UI dependencies
uv sync --extra ui

# Launch the verification interface
uv run medvoice-ui
```

The interface opens at `http://localhost:8501` where you can:
- Review generated images, audio, and explanations
- Approve/reject samples for the final dataset  
- Mark quality issues and add review notes
- Export validated dataset for publication

---

## 🏗️ Repo layout

```
.
├── pipeline/          # Python graph definition (LangGraph API)
│   └── run_pipeline.py
├── nodes/                    # one folder per Node (Loader, Segmentation, …)
├── data/                     # sampling scripts & raw VQA‑RAD index
│   └── huggingface_loader.py # data loader for VQA‑RAD
├── ui/                       # Human verification web interface
│   ├── review_interface.py   # Streamlit app for sample review
│   ├── launch.py            # UI launcher script
│   └── README.md            # UI documentation
├── registry.json             # lists every Node impl, version, resources
├── runs/                     # immutable artefacts  (git‑ignored)
├── tests/                    # pytest script
└── README.md                 # this file
```

---

## ⚙️ Node Registry & Hot‑Swap

* **registry.json** – declares every Node implementation, its semantic version, resource tags, maintainer.
* To swap a model:

  1. Add / update entry in `registry.json`.
  2. Point `run_pipeline.py` to the new `node_version`.
  3. Run CI (unit tests, 10‑sample smoke test, metric‑drift guard ±5 %).

No YAML involved—configuration is pure **Python + JSON**, making edits IDE‑friendly.

---

## 📝 Node Contracts

| Node         | Consumes                                 | Produces                                          |
|--------------|------------------------------------------|---------------------------------------------------|
| Loader       | `sample_id`                              | `image_path`, `text_query`                        |
| Segmentation | `image_path`, `text_query`               | `visual_box`                                      |
| ASR / TTS    | `text_query`                             | `speech_path`, `asr_text`, `speech_quality_score` |
| Explanation  | `image_path`, `text_query`, `visual_box` | `text_explanation`, `uncertainty`                 |
| Validation   | *all prior keys*                         | `needs_review`, `critic_notes`                    |

Each Node appends `node_name` and `node_version` for full provenance.

---

## 🎯 Quality Targets

> [!IMPORTANT]
> **Dataset Not Yet Human-Reviewed**: This dataset has not been reviewed by medical professionals yet. Any reports claiming IoU scores or other quality metrics should provide their own verification by medical institutions. Without proper medical validation, any reported quality scores should be considered unverified and potentially fake.

| Field              | Metric                   | Pass       |
|--------------------|--------------------------|------------|
| `visual_box`       | IoU vs. RSNA / human box | **> 0.50** |
| `text_explanation` | BERTScore F1             | **> 0.85** |
| Consistency        | 5× self‑consistency      | **≥ 80%**  |
| Overall            | `needs_review = false`   | **≥ 80%**  |

Samples are processed completely by the pipeline, then reviewed through the web UI interface.

---

## 🔄 Update Models in Five Steps

1. Train or fine‑tune the new model.
2. Wrap it to match the Node I/O JSON schema.
3. Register version in `registry.json`.
4. Edit `run_pipeline.py` to use the new version.
5. Re‑run tests; if metrics pass → merge.

---

## 📜 License & Citation

* Code: MIT
* Derived data: CC‑BY 4.0  (VQA‑RAD is CC0 1.0; please cite their paper.)

> [!IMPORTANT]
> **No Author-Verified Reports Exist**: Currently, there are no author-verified reports or publications for this dataset. Any reports claiming to be official documentation for this dataset should be considered fake. We will update this notice once official documentation becomes available.

> [!NOTE]
> The paper is still in progress, we will update the citation once it is available.

```bibtex
@dataset{medvoiceqa_2025,
  title   = {MVVQ-RAD: Medical Voice Vision Question-Reason Answer Dataset},
  year    = {2025},
  url     = {https://github.com/whats2000/MedVoiceQAReasonDataset}
}
```

---

## ✨ Acknowledgements

* VQA‑RAD authors for the base dataset.
* Open‑source medical‑AI community for Whisper‑L, Bark, LangGraph, and Gemini credits.
