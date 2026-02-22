---
title: HealthPost
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.31.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# HealthPost - Complete CHW Decision Support

**MedGemma Impact Challenge 2025**

HealthPost is a complete decision support tool for Community Health Workers (CHWs) that supports the entire patient visit workflow: **Intake → Diagnose → Prescribe → Dispense**.

## The Problem

Community Health Workers serve as BOTH primary doctor AND pharmacist for 80% of rural healthcare worldwide. They need a single tool that:
- Captures patient symptoms (voice-enabled for low literacy)
- Analyzes medical images (skin conditions, wounds)
- Provides diagnosis and treatment guidance
- Checks drug safety before dispensing

## The Solution

HealthPost combines MedGemma 1.5 capabilities into one seamless workflow:

```
PATIENT VISIT ─────────────────────────────────────────────────────►

  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  1. INTAKE  │───▶│ 2. DIAGNOSE │───▶│ 3. PRESCRIBE│───▶│ 4. DISPENSE │
  │             │    │             │    │             │    │             │
  │ Voice: sym- │    │ Photo: rash │    │ AI suggests │    │ Scan exist- │
  │ ptom descr. │    │ wound, eyes │    │ treatment   │    │ ing meds    │
  │ (MedASR)    │    │(MedGemma1.5)│    │ options     │    │ Check safety│
  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Features

- **Voice Symptom Capture**: Record patient descriptions using MedASR
- **Medical Image Analysis**: Analyze skin conditions, wounds, and eyes using MedGemma 1.5 Vision
- **AI-Powered Diagnosis**: Get diagnosis with confidence scores and differential diagnoses
- **Agentic Reasoning**: MedGemma 1.5 autonomously reasons through cases step-by-step (ReAct agent loop)
- **Treatment Recommendations**: Evidence-based treatment plans appropriate for CHW level
- **Drug Interaction Checking**: Offline database with 300+ essential medicines + DDInter API
- **Referral Guidance**: Know when to escalate to hospital
- **Works Offline**: Designed for edge deployment without internet

## Quick Start

### Prerequisites

HealthPost requires **MedGemma 1.5** via HuggingFace. You need:
- Python 3.10+
- `transformers` and `torch` installed
- Access to `google/medgemma-1.5-4b-it` on HuggingFace (gated model — see below)
- A CUDA GPU is recommended (4-bit quantization needs ~4 GB VRAM)

The app will **not** start without `transformers` and `torch` — there are no fallback backends.

### HuggingFace Authentication

MedGemma 1.5 is a **gated model**. You must request access and authenticate before using it:

1. **Create a HuggingFace account** at https://huggingface.co/join (if you don't have one)
2. **Request access** to the model at https://huggingface.co/google/medgemma-1.5-4b-it — accept Google's license terms
3. **Create an access token** at https://huggingface.co/settings/tokens (select `read` scope)
4. **Log in** from your terminal:

```bash
pip install huggingface_hub
huggingface-cli login
```

Paste your token when prompted. This stores it locally so `transformers` can download the model.

Alternatively, set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=hf_your_token_here   # Linux/macOS
set HF_TOKEN=hf_your_token_here      # Windows cmd
```

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

# Log in to HuggingFace (if not already done)
huggingface-cli login
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
!pip install gradio transformers torch bitsandbytes accelerate pillow soundfile huggingface_hub

# Authenticate with HuggingFace (use your token)
from huggingface_hub import login
login()  # Paste your HF token when prompted

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
├── config.py            # Configuration (MedGemma 1.5 settings)
├── core.py              # Main HealthPost orchestrator
├── agent.py             # ReAct agent for autonomous clinical reasoning
├── voice.py             # Voice transcription (MedASR)
├── vision.py            # Medical image analysis (MedGemma 1.5 Vision)
├── triage.py            # Diagnosis and treatment reasoning (MedGemma 1.5 Text)
├── drugs.py             # Drug database and interactions
├── ddinter_api.py       # DDInter drug interaction API client
└── rxnorm_api.py        # RxNorm API for drug name resolution

app.py                   # Gradio web interface
data/drugs.db            # SQLite drug database
requirements.txt         # Python dependencies
README.md                # This file
TECHNICAL_WRITEUP.md     # Detailed technical writeup
```

## Usage Guide

### Quick Workflow (Recommended)

1. Go to the **Quick Workflow** tab
2. Record or type patient symptoms
3. Upload any relevant medical images
4. List current medications (optional)
5. Enable **Agentic Workflow** for autonomous step-by-step reasoning
6. Click **Run Complete Workflow**
7. Review the comprehensive visit summary and AI reasoning trace

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
- Get alternative medication suggestions when interactions are found

## Technical Details

### Models Used

| Model | Task | Purpose |
|-------|------|---------|
| MedASR | Speech-to-text | Transcribe symptom descriptions |
| MedGemma 1.5 4B | Vision | Analyze medical images |
| MedGemma 1.5 4B | Vision | Extract text from prescriptions |
| MedGemma 1.5 4B | Reasoning | Generate diagnosis and treatment |
| MedGemma 1.5 4B | Agentic | Autonomous ReAct clinical reasoning |

### Drug Database

- **Coverage**: WHO Essential Medicines List (~300 drugs)
- **Format**: SQLite for offline use
- **Online enrichment**: DDInter API for additional interaction data
- **Data includes**: Drug info, classes, contraindications, common doses, interactions

### Edge Deployment

For mobile deployment:
- Uses 4-bit quantization to reduce model size (~4 GB VRAM)
- SQLite database works offline
- Gradio can be wrapped in Android WebView

## Safety Notice

> **This tool is designed to SUPPORT clinical decision-making, not replace it.**

- Always use clinical judgment
- Refer complex cases to higher levels of care
- Drug information should be verified
- Diagnoses are suggestions with confidence scores

## Evaluation Metrics

- Full workflow completes in < 30 seconds on T4 GPU
- Works completely offline after model download
- Covers common CHW scenarios: malaria, respiratory infections, skin conditions, wounds

## Contributing

This project was built for the MedGemma Impact Challenge. Contributions welcome!

## License

MIT License - See LICENSE file

## Acknowledgments

- Google for MedGemma 1.5 models
- WHO Essential Medicines List
- Community Health Workers worldwide who inspired this project

---

**Built for the MedGemma Impact Challenge 2025**
*Supporting the 3 billion people who rely on CHWs*
