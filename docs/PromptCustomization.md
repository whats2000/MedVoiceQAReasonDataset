# Prompt Customization Guide

The MedVoiceQA pipeline supports customizable prompts for all AI nodes. This allows you to fine-tune the behavior of each component without modifying code.

## Overview

The prompt management system:
- ✅ Loads custom prompts from `prompts/*.txt` files
- ✅ Validates prompts contain required variables  
- ✅ Falls back to default prompts if validation fails
- ✅ Provides warning messages for invalid prompts
- ✅ Caches prompts for performance

## Available Prompts

| Node | File | Required Variables | Purpose |
|------|------|-------------------|---------|
| Segmentation | `prompts/segmentation.txt` | `{text_query}` | Visual localization prompt for Gemini Vision |
| Explanation | `prompts/explanation.txt` | `{query}`, `{visual_box}` | Medical reasoning prompt for Gemini Language |
| Validation | `prompts/validation.txt` | `{query}`, `{visual_box}`, `{speech_quality}`, `{asr_text}`, `{explanation}`, `{uncertainty}` | Quality assessment prompt |

## Customizing Prompts

### 1. Edit Existing Prompts

Simply edit the files in the `prompts/` directory:

```bash
# Edit segmentation prompt
notepad prompts/segmentation.txt

# Edit explanation prompt  
notepad prompts/explanation.txt

# Edit validation prompt
notepad prompts/validation.txt
```

### 2. Variable Requirements

**⚠️ Important:** Your custom prompts MUST contain all required variables or they will be rejected.

For example, the segmentation prompt must include `{text_query}`:

```text
You are a medical AI. Analyze this image for: {text_query}

Please identify the relevant region...
```

### 3. Validation

The system automatically validates prompts when loaded:

- ✅ **Valid prompt**: Contains all required variables → Used
- ❌ **Invalid prompt**: Missing variables → Falls back to default + warning

### 4. Testing Custom Prompts

Run a single sample to test your changes:

```bash
uv run python pipeline/run_pipeline.py --limit 1
```

Check the logs for prompt validation messages:
- `Using custom prompt: segmentation` ✅ Success
- `Custom prompt 'segmentation' failed validation, using default` ❌ Failed

## Examples

### Custom Segmentation Prompt

```text
You are an expert radiologist. For the medical question: {text_query}

Identify the most clinically relevant region in this medical image.

Return your analysis as JSON:
{
  "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
  "confidence": 0.95,
  "region_description": "Left frontal lobe showing hypodense lesion",
  "relevance_reasoning": "This region directly answers the question about stroke location"
}
```

### Custom Explanation Prompt

```text
As a board-certified radiologist, analyze this medical image.

CLINICAL QUESTION: {query}
REGION OF INTEREST: {visual_box}

Provide your expert assessment:

1. IMAGING FINDINGS:
   - Describe what you observe
   - Note any abnormalities

2. CLINICAL CORRELATION:
   - How findings relate to the question
   - Differential diagnoses

3. FINAL ASSESSMENT:
   - Your conclusion
   - Confidence level

Format: REASONING: [your analysis]
UNCERTAINTY_SCORE: [0.0-1.0]
```

## Best Practices

1. **Keep Required Variables**: Always include all required `{variables}`
2. **Test Changes**: Run pipeline after modifications to verify prompts work
3. **Backup Originals**: Keep copies of working prompts before major changes
4. **Medical Accuracy**: Ensure prompts maintain medical terminology standards
5. **Clear Instructions**: Be specific about desired output format

## Troubleshooting

### Problem: "Custom prompt failed validation"

**Cause**: Missing required variables in your custom prompt

**Solution**: Check the required variables table above and ensure all are included

### Problem: "PromptManager not found" 

**Cause**: Import path issue

**Solution**: This is handled automatically by the system

### Problem: Pipeline uses default prompt despite custom file

**Cause**: Custom prompt file is empty or has syntax errors

**Solution**: 
1. Check file is not empty
2. Verify UTF-8 encoding
3. Ensure variables use correct `{variable}` syntax

## Advanced Usage

### Creating New Prompt Templates

Use the prompt manager programmatically:

```python
from prompts import PromptManager

pm = PromptManager()

# Create a template from current default
pm.create_template_prompt("my_custom_segmentation", "Default prompt with {text_query}")

# List available prompts
available = pm.list_available_prompts()
print(available)
```

### Programmatic Access

```python
from prompts import PromptManager

pm = PromptManager()

# Get a prompt with validation
prompt = pm.get_prompt(
    "segmentation",
    "fallback prompt", 
    {"text_query"}
)

# Clear cache if needed
pm.clear_cache()
```
