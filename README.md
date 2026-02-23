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

HealthPost combines MedGemma + MedASR capabilities into one seamless workflow:

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
- **Medical Image Analysis**: Analyze skin conditions, wounds, and eyes using MedGemma Vision
- **AI-Powered Diagnosis**: Get diagnosis with confidence scores and differential diagnoses
- **Structured Clinical Reasoning**: MedGemma generates validated JSON clinical assessments (diagnosis, treatment, referral)
- **Treatment Recommendations**: Evidence-based treatment plans appropriate for CHW level
- **Drug Interaction Checking**: DDInter API with 302K+ drug-drug interaction associations
- **Referral Guidance**: Know when to escalate to hospital
- **Edge Deployment**: Model inference runs locally; drug interaction checks use DDInter API when online

## Quick Start

### Prerequisites

HealthPost requires **MedGemma** via HuggingFace. You need:
- Python 3.10+
- `transformers` and `torch` installed
- Access to `google/medgemma-4b-it` on HuggingFace (gated model — see below)
- A CUDA GPU is recommended (4-bit quantization needs ~4 GB VRAM)

The app will **not** start without `transformers` and `torch` — there are no fallback backends.

### HuggingFace Authentication

MedGemma is a **gated model**. You must request access and authenticate before using it:

1. **Create a HuggingFace account** at https://huggingface.co/join (if you don't have one)
2. **Request access** to the model at https://huggingface.co/google/medgemma-4b-it — accept Google's license terms
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
git clone https://github.com/El-Moghazy2/gemma.git
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
!git clone https://github.com/El-Moghazy2/gemma.git
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
├── config.py            # Configuration (MedGemma settings)
├── core.py              # Main HealthPost orchestrator
├── visit_graph.py       # LangGraph StateGraph patient visit pipeline
├── voice.py             # Voice transcription (MedASR)
├── vision.py            # Medical image analysis (MedGemma Vision)
├── triage.py            # Diagnosis and treatment reasoning (MedGemma Text)
├── drugs.py             # Drug interaction checking (DDInter API)
├── ddinter_api.py       # DDInter drug interaction API client
└── rxnorm_api.py        # RxNorm API for drug name resolution

app.py                   # Gradio web interface
data/drugs.db            # Drug reference data
requirements.txt         # Python dependencies
README.md                # This file
TECHNICAL_WRITEUP.md     # Detailed technical writeup
```

## Usage Guide

### Clinical Workspace

1. Open the **Clinical Workspace** tab
2. Enter patient symptoms via text or record audio (MedASR transcription)
3. Optionally upload medical images (skin conditions, wounds, etc.)
4. Optionally enter current medications (text or photo of medication label)
5. Click **Run Complete Workflow** to get a full clinical assessment
6. Review the diagnosis, treatment plan, drug interaction checks, and referral guidance
7. Use the follow-up chat to ask clarifying questions about the assessment

## Technical Details

### Models Used

| Model | ID | Task | Purpose |
|-------|-----|------|---------|
| MedASR | `google/medasr` | Speech-to-text | Transcribe symptom descriptions |
| MedGemma 4B | `google/medgemma-4b-it` | Vision + text reasoning | Medical image analysis, diagnosis, treatment plans, structured JSON assessments |

### Drug Interaction Checking

- **Source**: DDInter API — 302,516 drug-drug interaction associations between 2,290 drugs
- **Data sources**: Compiled from DrugBank, KEGG, and other authoritative databases
- **Features**: Severity levels, clinical descriptions, management recommendations

### Edge Deployment

For mobile deployment:
- Uses 4-bit quantization to reduce model size (~4 GB VRAM)
- Model inference runs locally after initial download
- Gradio can be wrapped in Android WebView

## Safety Notice

> **This tool is designed to SUPPORT clinical decision-making, not replace it.**

- Always use clinical judgment
- Refer complex cases to higher levels of care
- Drug information should be verified
- Diagnoses are suggestions with confidence scores

## Evaluation Metrics

- Full workflow completes in < 30 seconds on T4 GPU
- Model inference works locally after initial download; drug interaction checks require internet
- Covers common CHW scenarios: malaria, respiratory infections, skin conditions, wounds

## Contributing

This project was built for the MedGemma Impact Challenge. Contributions welcome!

## License

MIT License - See LICENSE file

## References

- Sellergren, A. B., et al. "MedGemma: A Collection of Gemma-Based Models for Medical Applications." arXiv:2507.05201, 2025.
- Wu, C., et al. "LAST: Scalable Lattice-Based Speech Modelling in JAX." Proc. IEEE ICASSP, 2023.
- Google Health AI. "MedASR Model Card." https://developers.google.com/health-ai-developer-foundations/medasr/model-card
- Xiong, G., et al. "DDInter: An Online Drug–Drug Interaction Database." Nucleic Acids Research, 50(D1), 2022.
- LangChain, Inc. "LangGraph." https://github.com/langchain-ai/langgraph

## Acknowledgments

- Google for MedGemma and MedASR models
- DDInter drug interaction database
- Community Health Workers worldwide who inspired this project

---

**Built for the MedGemma Impact Challenge 2025**
*Supporting the 3 billion people who rely on CHWs*
