# MedVoiceQA Human Verification UI

A Streamlit-based web interface for reviewing and validating processed samples from the MedVoiceQA pipeline.

## Features

### 📊 **Dashboard Overview**
- Review progress tracking
- Quality statistics and metrics
- Issue categorization charts
- Approval rate monitoring

### 🔍 **Sample Review Interface**
- **Visual Review**: Display medical images with bounding boxes
- **Audio Playback**: Listen to generated speech with ASR verification
- **Content Validation**: Review AI-generated explanations and reasoning
- **Quality Metrics**: View uncertainty scores and quality assessments

### ✅ **Human Validation Tools**
- **Approval System**: Accept/reject samples for final dataset
- **Quality Rating**: 1-5 star rating system
- **Issue Categorization**: Mark specific problem types
- **Review Notes**: Add detailed comments and feedback

### 📁 **Data Management**
- **Run Selection**: Choose from available pipeline runs
- **Filtering**: Show flagged samples, unreviewed items
- **Sorting**: By uncertainty, quality score, review status
- **Export**: Download review decisions as JSON

## Quick Start

### 1. Install UI Dependencies

```bash
uv sync --extra ui
```

### 2. Launch the Interface

```bash
# Option 1: Using the launcher script
uv run medvoice-ui

# Option 2: Direct Streamlit launch
uv run streamlit run ui/review_interface.py
```

The interface will open automatically in your browser at `http://localhost:8501`

### 3. Review Workflow

1. **Select Run**: Choose a pipeline run from the sidebar
2. **Load Data**: Click "Load Run Data" to import processed samples
3. **Filter/Sort**: Use options to focus on specific sample types
4. **Review Samples**: 
   - Examine images, audio, and AI explanations
   - Rate quality and mark issues
   - Approve/reject for final dataset
5. **Save Progress**: Reviews are saved automatically
6. **Export Results**: Download final review decisions

## Interface Sections

### Sidebar Controls
- **Reviewer Name**: Identity tracking
- **Run Selection**: Choose data source
- **Progress Metrics**: Review completion stats
- **Export Options**: Download review data

### Main Content
- **Filter Bar**: Sample filtering and sorting
- **Sample Cards**: Individual review interfaces
- **Pagination**: Navigate through large datasets
- **Summary Stats**: Overall review statistics

## Sample Review Process

For each sample, reviewers evaluate:

### Visual Components
- ✅ **Image Quality**: Clear, properly formatted medical images
- ✅ **Bounding Boxes**: Accurate region localization
- ✅ **Visual Relevance**: Appropriate for the question asked

### Audio Components  
- ✅ **Speech Quality**: Clear, natural-sounding generated audio
- ✅ **ASR Accuracy**: Correct transcription of spoken text
- ✅ **Audio Clarity**: No artifacts or distortions

### Text Components
- ✅ **Explanation Quality**: Accurate, relevant medical reasoning
- ✅ **Language Fluency**: Clear, professional medical language
- ✅ **Factual Accuracy**: Correct medical information

### Overall Assessment
- ✅ **Question Relevance**: Response addresses the question
- ✅ **Multi-modal Coherence**: All components work together
- ✅ **Educational Value**: Suitable for training/research use

## Issue Categories

When marking problems, reviewers can select from:

- **Image Quality Issues**: Blurry, corrupted, or inappropriate images
- **Audio Quality Issues**: Poor synthesis, background noise, distortion
- **Text/Explanation Issues**: Inaccurate, unclear, or irrelevant text
- **Bounding Box Issues**: Incorrect localization or missing regions
- **Reasoning Issues**: Flawed logic, medical inaccuracies
- **Other Issues**: Additional problems not covered above

## Data Export

The interface exports review data including:

```json
{
  "run_id": "20240127_143022-a1b2c3d4",
  "export_timestamp": "2024-01-27T18:45:00Z",
  "reviewer": "dr_smith",
  "stats": {
    "total_samples": 300,
    "reviewed_samples": 250,
    "approved_samples": 220,
    "approval_rate": 0.88
  },
  "reviews": {
    "sample_001": {
      "approved": true,
      "quality_rating": 4,
      "notes": "Good quality, minor audio issue",
      "issues": {
        "audio_quality": true,
        "other": false
      },
      "reviewed_at": "2024-01-27T18:30:00Z"
    }
  }
}
```

## Configuration

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Port for the web interface (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)

### Customization
- Modify `ui/review_interface.py` to adjust the interface
- Add custom validation rules in the reviewer class
- Extend issue categories for specific use cases

## Tips for Effective Review

1. **Start with Flagged Samples**: Focus on automatically identified issues first
2. **Use Filtering**: Review similar sample types together for consistency  
3. **Take Breaks**: Medical content review can be cognitively demanding
4. **Document Issues**: Use detailed notes for complex problems
5. **Batch Similar Decisions**: Group approvals/rejections for efficiency

## Troubleshooting

### Common Issues

**"No pipeline runs found"**
- Ensure you've run the pipeline first: `uv run python pipeline/run_pipeline.py`
- Check that `runs/` directory exists with valid data

**"Error loading image"**
- Verify image paths in pipeline results
- Check file permissions and disk space

**"Streamlit not found"**
- Install UI dependencies: `uv sync --extra ui`

**Interface won't load**
- Check port availability (8501)
- Try different browser or clear cache

### Performance Notes

- Large datasets (>100 samples) may load slowly
- Use pagination to improve responsiveness
- Audio files require browser codec support
- Images are loaded on-demand for efficiency

## Integration with Pipeline

The UI automatically discovers pipeline runs from the `runs/` directory:

```
runs/
├── 20240127_143022-a1b2c3d4/
│   ├── results.json          # Pipeline output data
│   ├── manifest.json         # Run configuration
│   ├── pipeline.log         # Processing logs
│   └── human_review.json    # UI saves here
```

Review data is saved alongside pipeline results for full provenance tracking.
