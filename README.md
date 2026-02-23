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

# HealthPost -- CHW Decision Support System

**MedGemma Impact Challenge 2025**

HealthPost is a complete clinical decision-support tool for Community Health
Workers (CHWs). It covers the entire patient visit workflow in a single
interface: **Intake -- Diagnose -- Prescribe -- Dispense**.

```
PATIENT VISIT ──────────────────────────────────────────────────────────►

  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  1. INTAKE   │───▶│ 2. DIAGNOSE  │───▶│ 3. PRESCRIBE │───▶│ 4. DISPENSE  │
  │              │    │              │    │              │    │              │
  │ Voice/text   │    │ Photo: rash, │    │ AI treatment │    │ Scan meds,   │
  │ symptoms     │    │ wound, eyes  │    │ plan + meds  │    │ check safety │
  │ (MedASR)     │    │ (MedGemma)   │    │ (MedGemma)   │    │ (DDInter)    │
  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Table of Contents

1. [Features](#features)
2. [Live Demo](#live-demo)
3. [Reproducibility Guide](#reproducibility-guide)
   - [Prerequisites](#prerequisites)
   - [Hardware Requirements](#hardware-requirements)
   - [Step 1 -- HuggingFace Authentication](#step-1----huggingface-authentication)
   - [Step 2 -- Clone and Install](#step-2----clone-and-install)
   - [Step 3 -- Run Locally](#step-3----run-locally)
   - [Step 4 -- Verify the Workflow](#step-4----verify-the-workflow)
   - [Kaggle Notebook Reproduction](#kaggle-notebook-reproduction)
   - [HuggingFace Spaces Deployment](#huggingface-spaces-deployment)
4. [Architecture](#architecture)
5. [Project Structure](#project-structure)
6. [Configuration Reference](#configuration-reference)
7. [Dependency Pinning](#dependency-pinning)
8. [Troubleshooting](#troubleshooting)
9. [Usage Guide](#usage-guide)
10. [Safety Notice](#safety-notice)
11. [References](#references)
12. [License](#license)

---

## Features

- **Voice Symptom Capture** -- Record patient descriptions using Google MedASR
  (with Whisper fallback).
- **Medical Image Analysis** -- Analyze skin conditions, wounds, and eyes via
  MedGemma Vision.
- **AI-Powered Diagnosis** -- Structured JSON clinical assessments with
  confidence scores and differential diagnoses.
- **Treatment Recommendations** -- Evidence-based treatment plans appropriate
  for CHW level, including medications, instructions, and warning signs.
- **Drug Interaction Checking** -- DDInter API with 302K+ drug-drug interaction
  associations compiled from DrugBank, KEGG, and other authoritative sources.
- **Referral Guidance** -- Rule-based escalation when conditions exceed CHW
  scope, or when severe drug interactions are detected.
- **Follow-up Chat** -- After a diagnosis, the CHW can ask clarifying questions
  in a contextual chat grounded in the visit result.
- **One-Click Workflow** -- All modules run automatically in sequence via a
  LangGraph `StateGraph` pipeline.

## Live Demo

The hosted version is available at:

> **https://huggingface.co/spaces/ElMoghazy/healthpost**

No installation required. Runs on HuggingFace ZeroGPU (NVIDIA T4).

---

## Reproducibility Guide

This section explains exactly how to reproduce the project from scratch on a
clean machine, on Kaggle, or on HuggingFace Spaces.

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 -- 3.12 | 3.11 recommended (matches Spaces) |
| pip | >= 23.0 | Needed for URL-based requirements |
| Git | >= 2.25 | For cloning the repo |
| CUDA toolkit | >= 11.8 | Only for local GPU inference |
| Internet | Required | For model download and DDInter API |

### Hardware Requirements

| Environment | GPU | VRAM | Disk | RAM | Notes |
|-------------|-----|------|------|-----|-------|
| **Local (recommended)** | NVIDIA GPU | >= 6 GB | ~12 GB | >= 16 GB | 4-bit quantization fits in 6 GB |
| **Local (CPU-only)** | None | N/A | ~12 GB | >= 32 GB | Very slow; for testing only |
| **Kaggle** | T4 (free tier) | 16 GB | 20 GB | 13 GB | Use GPU accelerator |
| **HF Spaces (ZeroGPU)** | T4 (shared) | 16 GB | 50 GB | 16 GB | Free tier works |

Model disk footprint:
- `google/medgemma-4b-it`: ~8 GB (downloaded once, cached)
- `google/medasr`: ~1.2 GB
- Total Python environment: ~3 GB

### Step 1 -- HuggingFace Authentication

MedGemma is a **gated model**. You must request access before the weights can
be downloaded.

1. **Create a HuggingFace account** at https://huggingface.co/join (if you do
   not have one).

2. **Request model access** -- visit each model page and accept the license:
   - https://huggingface.co/google/medgemma-4b-it (Google's license terms)
   - https://huggingface.co/google/medasr (Google's license terms)

   Approval is typically instant.

3. **Create an access token** at https://huggingface.co/settings/tokens.
   Select the **Read** scope.

4. **Authenticate locally** (choose one method):

   ```bash
   # Method A: interactive login (stores token in ~/.cache/huggingface/)
   pip install huggingface_hub
   huggingface-cli login
   # Paste your token when prompted.

   # Method B: environment variable (useful for CI/scripts)
   export HF_TOKEN=hf_your_token_here       # Linux / macOS
   set HF_TOKEN=hf_your_token_here          # Windows cmd
   $env:HF_TOKEN = "hf_your_token_here"     # Windows PowerShell
   ```

### Step 2 -- Clone and Install

```bash
# Clone the repository
git clone https://github.com/El-Moghazy2/gemma.git
cd gemma

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows cmd
# venv\Scripts\Activate.ps1       # Windows PowerShell

# Install all dependencies
pip install -r requirements.txt
```

**Important note on `transformers`**: The `requirements.txt` pins a specific
commit of `transformers` (`65dc261`) via a GitHub archive URL. This is required
because MedASR produces corrupted `<epsilon>` tokens on newer releases. Do
**not** override this with a PyPI install.

If installation fails on `bitsandbytes` (common on Windows or non-CUDA
systems), you can install the CPU-compatible build:

```bash
pip install bitsandbytes --prefer-binary
```

Or skip it entirely -- the app will fall back to full-precision loading (needs
more VRAM).

### Step 3 -- Run Locally

```bash
python app.py
```

Expected console output:

```
==================================================
HealthPost - CHW Decision Support System
==================================================

Pre-loading models...
INFO:healthpost.core:HealthPost instance created
INFO:healthpost.core:Initializing HealthPost components...
INFO:healthpost.core:Drug database loaded
INFO:healthpost.voice:Loading MedASR (google/medasr)...
INFO:healthpost.voice:MedASR loaded successfully
INFO:healthpost.inference_backend:Loading model google/medgemma-4b-it ...
INFO:healthpost.inference_backend:4-bit quantization enabled
INFO:healthpost.inference_backend:Model loaded successfully
...
Models ready!

Starting application...
Running on local URL: http://0.0.0.0:7860
```

Open your browser to **http://localhost:7860**.

The first run downloads `google/medgemma-4b-it` (~8 GB) and `google/medasr`
(~1.2 GB). Subsequent runs use the cached weights.

To share via a public Gradio URL:

```bash
python app.py --share
```

### Step 4 -- Verify the Workflow

Use the built-in demo scenarios to verify that every pipeline stage works:

1. Open the **Clinical Workspace** tab.
2. Select **"Malaria Case"** from the demo dropdown. This auto-fills symptoms,
   age, and current medications.
3. Click **Run Complete Workflow**.
4. Wait for the pipeline to finish (~20-30s on T4 GPU). You should see:
   - **Diagnosis**: A condition with confidence score and differential
     diagnoses.
   - **Treatment Plan**: Medications with dosages, instructions, and warning
     signs.
   - **Drug Safety Check**: Interaction results for the pre-filled medications
     (Metformin + Amlodipine + any proposed meds).
   - **Referral Guidance**: Whether the patient needs hospital referral.
5. After the report appears, a **Follow-up Questions** chat section becomes
   visible. Type a question like "What if the patient is pregnant?" to verify
   the contextual chat works.

Repeat with the other demo scenarios to exercise different code paths:

| Scenario | Tests |
|----------|-------|
| Malaria Case | Diagnosis + drug interactions (Metformin + Amlodipine) |
| Skin Condition | Vision analysis (if image provided) + pediatric dosing |
| Wound + Drug Interaction | Severe interaction detection (Warfarin + Metformin) + alternative suggestion |
| Child Diarrhea | Pediatric case + supportive care (no medications) |

### Kaggle Notebook Reproduction

For the MedGemma Impact Challenge submission, use a Kaggle notebook with GPU:

```python
# 1. Install dependencies (run in first cell)
!pip install -q gradio torch accelerate bitsandbytes sentencepiece \
    protobuf pydantic Pillow numpy langgraph langchain-core librosa \
    huggingface_hub spaces
!pip install -q "transformers @ https://github.com/huggingface/transformers/archive/65dc261512cbdb1ee72b88ae5b222f2605aad8e5.zip"

# 2. Authenticate with HuggingFace
from huggingface_hub import login
login()  # Paste your HF token when prompted

# 3. Clone the repo
!git clone https://github.com/El-Moghazy2/gemma.git
%cd gemma

# 4. Launch the app
from app import create_interface
app = create_interface()
app.launch(share=True)  # share=True gives a public URL
```

**Kaggle-specific notes**:
- Select **GPU T4 x2** as the accelerator in notebook settings.
- The `share=True` flag creates a temporary public URL (valid ~72 hours).
- First run takes 5-10 minutes for model download.
- Kaggle sessions time out after 12 hours of inactivity.

### HuggingFace Spaces Deployment

The live demo at https://huggingface.co/spaces/ElMoghazy/healthpost runs on
HuggingFace ZeroGPU. To deploy your own copy:

1. **Create a new Space** at https://huggingface.co/new-space:
   - SDK: **Gradio**
   - Hardware: **ZeroGPU** (free tier) or **T4 small**

2. **Add your HF token as a secret**:
   - Go to Space Settings > Repository Secrets
   - Add `HF_TOKEN` with your token value

3. **Push the code**:

   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
   git push space master:main
   ```

4. The Space will build and start automatically. Monitor logs in the Space UI.

**ZeroGPU considerations**:
- The `@spaces.GPU(duration=300)` decorator in `app.py` requests GPU time
  per inference call. ZeroGPU assigns a shared T4 dynamically.
- Models must load on CPU first. The `Config.detect_device()` function returns
  `"cpu"` when `SPACE_ID` is set, and ZeroGPU handles GPU transfer
  transparently.
- The `HealthPost` object contains a LangGraph `CompiledStateGraph` with
  unpicklable closures. Therefore, `@spaces.GPU` functions must **not** receive
  `HealthPost` as an argument -- they call `get_healthpost()` inside the
  function body instead.

---

## Architecture

HealthPost uses a **LangGraph `StateGraph`** to orchestrate the patient visit
pipeline. Each step is a graph node with conditional routing:

```
                    ┌──────────────────┐
                    │      INTAKE      │
                    │ (voice / text)   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
             ┌──────│  route_after_     │──────┐
             │      │  intake           │      │
             │      └──────────────────┘      │
      (has images)                      (no images)
             │                                │
    ┌────────▼─────────┐                      │
    │  ANALYZE_IMAGES   │                      │
    │  (MedGemma Vision)│                      │
    └────────┬─────────┘                      │
             │                                │
             └───────────────┬────────────────┘
                    ┌────────▼─────────┐
                    │   EXTRACT_MEDS   │
                    │  (photo + text)  │
                    └────────┬─────────┘
                    ┌────────▼─────────┐
                    │     DIAGNOSE     │
                    │  (MedGemma Text) │
                    └────────┬─────────┘
                    ┌────────▼─────────┐
                    │   CHECK_DRUGS    │
                    │  (DDInter API)   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
             ┌──────│  route_after_     │──────┐
             │      │  drugs            │      │
             │      └──────────────────┘      │
      (has interactions)              (no interactions)
             │                                │
    ┌────────▼─────────┐                      │
    │ FIND_ALTERNATIVES │                      │
    │  (MedGemma Text)  │                      │
    └────────┬─────────┘                      │
             │                                │
             └───────────────┬────────────────┘
                    ┌────────▼─────────┐
                    │  ASSESS_SAFETY   │
                    │ (referral rules) │
                    └────────┬─────────┘
                             │
                            END
```

### Models Used

| Model | HF ID | Task | Purpose |
|-------|-------|------|---------|
| MedGemma 4B | `google/medgemma-4b-it` | Vision + text | Image analysis, diagnosis, treatment, structured JSON, follow-up chat |
| MedASR | `google/medasr` | Speech-to-text | Transcribe symptom descriptions |
| Whisper Small | `openai/whisper-small` | Speech-to-text | Fallback if MedASR fails to load |

### External APIs

| Service | Purpose | Rate Limit |
|---------|---------|------------|
| DDInter API (`ddinter2.scbdd.com`) | Drug-drug interaction checking | 5 req/s (self-throttled) |
| RxNorm API (`rxnav.nlm.nih.gov`) | Drug name resolution (backup) | 20 req/s |

---

## Project Structure

```
gemma/
├── app.py                   # Gradio web interface (main entry point)
├── requirements.txt         # Pinned Python dependencies
├── README.md                # This file
├── TECHNICAL_WRITEUP.md     # Detailed technical writeup
├── TECHNICAL_WRITEUP.pdf    # PDF version for submission
├── data/
│   └── drugs.db             # SQLite drug reference data (60 KB)
└── healthpost/
    ├── __init__.py           # Package exports (v0.3.0)
    ├── config.py             # Config dataclass + device detection
    ├── core.py               # HealthPost orchestrator + PatientVisitResult
    ├── visit_graph.py        # LangGraph StateGraph pipeline
    ├── inference_backend.py  # TransformersBackend / OllamaBackend
    ├── voice.py              # MedASR / Whisper transcription
    ├── vision.py             # MedGemma Vision image analysis
    ├── triage.py             # Diagnosis + treatment reasoning
    ├── drugs.py              # DrugDatabase (DDInter wrapper)
    ├── ddinter_api.py        # DDInter HTTP client
    └── rxnorm_api.py         # RxNorm HTTP client (backup)
```

---

## Configuration Reference

All configuration lives in `healthpost/config.py` as a `@dataclass`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hf_model_id` | `google/medgemma-4b-it` | HuggingFace model for text and vision inference |
| `medasr_model_id` | `google/medasr` | HuggingFace model for speech-to-text |
| `device` | Auto-detected | `"cuda"` locally, `"cpu"` on HF Spaces (ZeroGPU handles GPU) |
| `hf_use_4bit` | `True` | Enable 4-bit quantization via `bitsandbytes` |
| `max_new_tokens` | `512` | Maximum tokens per generation call |
| `temperature` | `0.3` | Sampling temperature for inference |
| `confidence_threshold` | `0.7` | Diagnosis confidence below this triggers referral |
| `sample_rate` | `16000` | Expected audio sample rate in Hz |
| `data_dir` | `./data` | Directory for static data assets |

To customize, edit `Config()` in `app.py` or create a config programmatically:

```python
from healthpost import Config, HealthPost

config = Config(
    hf_model_id="google/medgemma-4b-it",
    hf_use_4bit=False,          # Full precision (needs more VRAM)
    temperature=0.2,            # More deterministic
    confidence_threshold=0.6,   # Less conservative referrals
)
hp = HealthPost(config)
hp.warmup()
```

---

## Dependency Pinning

The `requirements.txt` pins a specific `transformers` commit to avoid a
known MedASR bug:

```
transformers @ https://github.com/huggingface/transformers/archive/65dc261512cbdb1ee72b88ae5b222f2605aad8e5.zip
```

This commit (`65dc261`) is required because later versions introduce an
`<epsilon>` token corruption bug that breaks MedASR transcription output.

All other dependencies use minimum version constraints (`>=`). The tested
combination as of February 2026:

| Package | Tested Version | Constraint |
|---------|---------------|------------|
| `torch` | 2.5.x | `>= 2.1.0` |
| `transformers` | commit `65dc261` | Pinned (see above) |
| `accelerate` | 1.2.x | `>= 0.25.0` |
| `bitsandbytes` | 0.45.x | `>= 0.41.0` |
| `gradio` | 5.31.x | `>= 4.0.0` |
| `pydantic` | 2.10.x | `>= 2.0.0` |
| `langgraph` | 0.3.x | `>= 0.2.0` |
| `langchain-core` | 0.3.x | `>= 0.2.0` |
| `Pillow` | 11.x | `>= 10.0.0` |
| `librosa` | 0.10.x | `>= 0.10.0` |
| `sentencepiece` | 0.2.x | `>= 0.2.0` |
| `protobuf` | 5.x | `>= 3.20.0` |
| `spaces` | 0.34.x | `>= 0.30.0` |

---

## Troubleshooting

### "Access to model google/medgemma-4b-it is restricted"

You have not been granted access to the gated model. Visit
https://huggingface.co/google/medgemma-4b-it and accept the license.
Then make sure you are authenticated (`huggingface-cli login` or `HF_TOKEN`).

### "No module named 'bitsandbytes'"

`bitsandbytes` requires a CUDA-capable system. On CPU-only machines:

```bash
pip install bitsandbytes --prefer-binary
```

If that also fails, the app will still work but loads the full-precision model
(needs ~16 GB RAM).

### MedASR produces garbled output with `<epsilon>` tokens

You are using an incompatible `transformers` version. Reinstall the pinned
commit:

```bash
pip install "transformers @ https://github.com/huggingface/transformers/archive/65dc261512cbdb1ee72b88ae5b222f2605aad8e5.zip"
```

### "CUDA out of memory"

MedGemma 4B with 4-bit quantization needs approximately 6 GB VRAM. If you have
less, try:
- Close other GPU processes
- Reduce `max_new_tokens` in `Config`
- Use CPU mode (`Config(device="cpu")`) -- very slow but works

### DDInter API timeout or connection error

The DDInter API (`ddinter2.scbdd.com`) is an external service. If it is down:
- The app continues to work -- drug interaction checks return empty results
- A warning is logged: `"DDInter client unavailable"`
- All other pipeline stages (diagnosis, treatment, referral) are unaffected

### App starts but model download is slow

First run downloads ~9 GB of model weights. On slow connections:
- The download is resumable -- if interrupted, restart the app and it picks up
  where it left off
- Models are cached in `~/.cache/huggingface/` (or `$HF_HOME`)
- On HF Spaces, the cache is persisted to `/data/.huggingface/` to survive
  container restarts

---

## Usage Guide

### Clinical Workspace

1. Open the **Clinical Workspace** tab.
2. Enter patient symptoms via text or record audio (MedASR transcription).
3. Optionally upload medical images (skin conditions, wounds).
4. Optionally enter current medications (text or photo of medication labels).
5. Click **Run Complete Workflow** to get a full clinical assessment.
6. Review the diagnosis, treatment plan, drug interaction checks, and referral
   guidance in the generated report.
7. Use the **Follow-up Questions** chat to ask clarifying questions about the
   assessment (e.g. "What if the patient is pregnant?", "Can I substitute this
   medication?").

### Architecture and About

The second tab provides an overview of the system architecture, module table,
and the medical disclaimer.

---

## Safety Notice

> **This tool is designed to SUPPORT clinical decision-making, not replace it.**

- Always apply clinical judgment alongside AI recommendations.
- Refer complex cases to higher levels of care.
- Drug interaction information should be verified with a pharmacist.
- Diagnoses are suggestions with confidence scores -- not definitive.
- All AI-generated recommendations should be reviewed by a qualified clinician.

---

## References

- Sellergren, A. B., et al. "MedGemma: A Collection of Gemma-Based Models for
  Medical Applications." arXiv:2507.05201, 2025.
- Wu, C., et al. "LAST: Scalable Lattice-Based Speech Modelling in JAX." Proc.
  IEEE ICASSP, 2023.
- Google Health AI. "MedASR Model Card."
  https://developers.google.com/health-ai-developer-foundations/medasr/model-card
- Xiong, G., et al. "DDInter: An Online Drug-Drug Interaction Database."
  Nucleic Acids Research, 50(D1), 2022.
- LangChain, Inc. "LangGraph." https://github.com/langchain-ai/langgraph

## License

MIT License -- See LICENSE file.

---

**Built for the MedGemma Impact Challenge 2025**
*Supporting the 3 billion people who rely on Community Health Workers*
