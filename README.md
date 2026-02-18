# HealthPost - Complete CHW Decision Support

**MedGemma Impact Challenge 2024**

HealthPost is a complete decision support tool for Community Health Workers (CHWs) that supports the entire patient visit workflow: **Intake → Diagnose → Prescribe → Dispense**.

## The Problem

Community Health Workers serve as BOTH primary doctor AND pharmacist for 80% of rural healthcare worldwide. They need a single tool that:
- Captures patient symptoms (voice-enabled for low literacy)
- Analyzes medical images (skin conditions, wounds)
- Provides diagnosis and treatment guidance
- Checks drug safety before dispensing

## The Solution

HealthPost combines four MedGemma capabilities into one seamless workflow:

```
PATIENT VISIT ─────────────────────────────────────────────────────►

  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  1. INTAKE  │───▶│ 2. DIAGNOSE │───▶│ 3. PRESCRIBE│───▶│ 4. DISPENSE │
  │             │    │             │    │             │    │             │
  │ Voice: sym- │    │ Photo: rash │    │ AI suggests │    │ Scan exist- │
  │ ptom descr. │    │ wound, eyes │    │ treatment   │    │ ing meds    │
  │ (MedASR)    │    │ (MedGemma)  │    │ options     │    │ Check safety│
  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Features

- **Voice Symptom Capture**: Record patient descriptions using MedASR
- **Medical Image Analysis**: Analyze skin conditions, wounds, and eyes using MedGemma vision
- **AI-Powered Diagnosis**: Get diagnosis with confidence scores and differential diagnoses
- **Treatment Recommendations**: Evidence-based treatment plans appropriate for CHW level
- **Drug Interaction Checking**: Offline database with 300+ essential medicines
- **Referral Guidance**: Know when to escalate to hospital
- **Works Offline**: Designed for edge deployment without internet

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/healthpost.git
cd healthpost

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

Open your browser to `http://localhost:7860`

### Kaggle Notebook

For the MedGemma Impact Challenge, use this in a Kaggle notebook:

```python
# Install dependencies
!pip install gradio transformers torch bitsandbytes accelerate pillow soundfile

# Clone the repo
!git clone https://github.com/your-repo/healthpost.git
%cd healthpost

# Run the app
from app import create_interface
app = create_interface()
app.launch(share=True)
```

## Project Structure

```
healthpost/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── core.py              # Main HealthPost orchestrator
├── voice.py             # Voice transcription (MedASR)
├── vision.py            # Medical image analysis (MedGemma)
├── drugs.py             # Drug database and interactions
└── triage.py            # Diagnosis and treatment reasoning

app.py                   # Gradio web interface
requirements.txt         # Python dependencies
README.md               # This file
```

## Usage Guide

### Quick Workflow (Recommended)

1. Go to the **Quick Workflow** tab
2. Record or type patient symptoms
3. Upload any relevant medical images
4. List current medications (optional)
5. Click **Run Complete Workflow**
6. Review the comprehensive visit summary

### Step-by-Step Workflow

For more control, use the **Step-by-Step** tab:

1. **INTAKE**: Record voice or type symptoms
2. **DIAGNOSE**: Upload and analyze medical images
3. **PRESCRIBE**: Generate diagnosis and treatment plan
4. **DISPENSE**: Check drug interactions before dispensing

### Drug Reference

Use the **Drug Reference** tab to:
- Look up drug information
- Check interactions between any medications

## Technical Details

### Models Used

| Model | Task | Purpose |
|-------|------|---------|
| MedASR | Speech-to-text | Transcribe symptom descriptions |
| MedGemma 4B | Vision | Analyze medical images |
| MedGemma 4B | Vision | Extract text from prescriptions |
| MedGemma 4B | Reasoning | Generate diagnosis and treatment |

### Drug Database

- **Coverage**: WHO Essential Medicines List (~300 drugs)
- **Format**: SQLite for offline use
- **Size**: ~50MB
- **Data includes**: Drug info, classes, contraindications, common doses, interactions

### Edge Deployment

For mobile deployment:
- Uses 4-bit quantization to reduce model size
- SQLite database works offline
- Gradio can be wrapped in Android WebView

## Safety Notice

⚠️ **This tool is designed to SUPPORT clinical decision-making, not replace it.**

- Always use clinical judgment
- Refer complex cases to higher levels of care
- Drug information should be verified
- Diagnoses are suggestions with confidence scores

## Evaluation Metrics

- Full workflow completes in < 30 seconds
- Works completely offline (airplane mode)
- Covers common CHW scenarios: malaria, respiratory infections, skin conditions, wounds

## Contributing

This project was built for the MedGemma Impact Challenge. Contributions welcome!

## License

MIT License - See LICENSE file

## Acknowledgments

- Google for MedGemma models
- WHO Essential Medicines List
- Community Health Workers worldwide who inspired this project

---

**Built for the MedGemma Impact Challenge 2024**
*Supporting the 3 billion people who rely on CHWs*
