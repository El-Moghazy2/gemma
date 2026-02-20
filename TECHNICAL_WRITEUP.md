# HealthPost: Complete CHW Decision Support

## MedGemma Impact Challenge - Technical Writeup

---

## 1. Problem Statement

### The Challenge

Community Health Workers (CHWs) serve as the primary healthcare providers for approximately **3 billion people** in low-resource settings worldwide. Unlike physicians in well-equipped facilities, CHWs face a unique challenge: they must act as **both doctor AND pharmacist** during patient visits.

Current digital health tools fail CHWs because they address only part of the workflow:
- Symptom checkers provide diagnosis but not treatment
- Drug reference apps check interactions but don't diagnose
- No tool supports the complete visit from intake to dispensing

### The Consequences

This gap leads to:
- **Misdiagnosis** due to limited training and no decision support
- **Medication errors** - the 3rd leading cause of death globally
- **Delayed referrals** when serious conditions go unrecognized
- **Drug interactions** when current medications aren't considered

### Our Solution

**HealthPost** is a complete decision support tool that guides CHWs through the entire patient visit:

```
INTAKE → DIAGNOSE → PRESCRIBE → DISPENSE
```

By integrating MedGemma 1.5's medical AI capabilities with an offline drug database, HealthPost provides:
1. Voice-based symptom capture using MedASR
2. Visual analysis of skin conditions and wounds
3. AI-powered diagnosis with agentic reasoning
4. Treatment recommendations appropriate for health post level
5. Drug interaction checking before dispensing

---

## 2. Technical Implementation

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    HEALTHPOST                            │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   INTAKE    │  │  DIAGNOSE   │  │  PRESCRIBE  │     │
│  │             │  │             │  │             │     │
│  │  MedASR     │─▶│  MedGemma   │─▶│  MedGemma   │     │
│  │  (voice)    │  │  1.5 Vision │  │  1.5 Text   │     │
│  │             │  │             │  │             │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                            │            │
│                          ┌─────────┐       │            │
│                          │  ReAct  │◀──────┘            │
│                          │  Agent  │                    │
│                          └────┬────┘                    │
│                               ▼                         │
│                       ┌─────────────┐                   │
│                       │  DISPENSE   │                   │
│                       │             │                   │
│                       │  Drug DB    │                   │
│                       │  + DDInter  │                   │
│                       │             │                   │
│                       └─────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

### Models Used

| Model | Capability | Application |
|-------|------------|-------------|
| **MedGemma 1.5 4B** | Medical Vision | Analyze skin conditions, wounds, rashes |
| **MedGemma 1.5 4B** | Medical Reasoning | Generate diagnosis and treatment plans |
| **MedGemma 1.5 4B** | Agentic Reasoning | Autonomous ReAct clinical decision loop |
| **MedASR** | Speech-to-Text | Transcribe patient symptom descriptions |

### Why MedGemma 1.5

We use `google/medgemma-1.5-4b-it` — the latest MedGemma release — which offers significant improvements over v1:

- **+5% on MedQA** (medical question answering benchmark)
- **+22% on EHRQA** (electronic health record question answering)
- **Improved medical imaging** accuracy
- **Same hardware footprint** (~4 GB VRAM with 4-bit quantization)

HealthPost requires MedGemma 1.5 and does not include fallback backends. If the model cannot be loaded, the application fails at startup with a clear error message. This ensures CHWs always receive the highest-quality AI assistance.

### Key Components

**1. Voice Symptom Capture (MedASR)**

MedASR transcribes spoken symptom descriptions with medical vocabulary awareness:
- Optimized for medical terminology
- Works with accented speech common in community health settings
- Feeds directly into the diagnosis pipeline

**2. Medical Image Analysis (MedGemma 1.5 Vision)**

MedGemma 1.5 Vision analyzes uploaded photos with condition-specific prompts:
- Skin conditions: Identifies rashes, lesions, infections
- Wounds: Assesses type, infection signs, healing stage
- Medication labels: Extracts drug names from photos of bottles/packages
- Provides severity assessment and recommended actions

Uses `AutoModelForImageTextToText` with `bfloat16` precision for optimal accuracy.

**3. Diagnosis Engine (MedGemma 1.5 Text)**

MedGemma 1.5 Text processes symptoms and visual findings to generate:
- Primary diagnosis with confidence level
- Differential diagnoses to consider
- Treatment recommendations appropriate for CHW level
- Referral guidance when hospital care is needed

Uses `AutoModelForImageTextToText` with `AutoProcessor` and `bfloat16` precision.

**4. Agentic Reasoning (ReAct Agent)**

HealthPost uses a **ReAct (Reason + Act)** agent loop for autonomous clinical reasoning:

1. MedGemma 1.5 receives the patient case
2. It **reasons** about what information is needed (`[THOUGHT]`)
3. It **acts** by calling tools like `analyze_skin`, `check_interactions` (`[ACTION]`)
4. It reviews observations and decides next steps (`[OBSERVATION]`)
5. After gathering enough information, it provides a complete assessment (`[FINAL_ANSWER]`)

This transparent reasoning builds CHW trust — they can see *why* the AI made each decision.

**5. Drug Safety Module**

Offline SQLite database containing:
- WHO Essential Medicines List (~300 drugs)
- Known drug-drug interactions with severity ratings
- Contraindications and dosing guidance

Supplemented by the **DDInter API** for additional interaction data when online.

Before dispensing, the system checks for interactions between:
- Patient's current medications
- Newly recommended medications

When severe interactions are detected, the system suggests alternative medications.

**6. Edge Deployment**

Designed for offline operation:
- 4-bit quantization reduces VRAM to ~4 GB (runs on consumer GPUs / Kaggle T4)
- SQLite database requires no internet connection
- Gradio UI works on mobile browsers

### Authentication

MedGemma 1.5 is a **gated model** on HuggingFace. To use it:

1. Request access at https://huggingface.co/google/medgemma-1.5-4b-it
2. Generate a token at https://huggingface.co/settings/tokens
3. Authenticate: `huggingface-cli login`

### Technical Specifications

```
Model:      MedGemma 1.5 4B (google/medgemma-1.5-4b-it)
Precision:  bfloat16 (or 4-bit NF4 quantized on CUDA)
Memory:     ~4 GB VRAM with 4-bit quantization
Inference:  <10 seconds per diagnosis on T4 GPU
Database:   ~5 MB SQLite (300 drugs, 50+ interactions)
Interface:  Gradio web UI (mobile-compatible)
Backend:    HuggingFace Transformers (no fallbacks)
Auth:       HuggingFace token required (gated model)
```

---

## 3. Evaluation & Impact

### Test Scenarios

We validated HealthPost against common CHW scenarios:

| Scenario | Symptoms | Expected | HealthPost Result |
|----------|----------|----------|-------------------|
| Malaria | Fever 3 days, headache, chills | Antimalarial + supportive | Correct |
| Gastroenteritis | Diarrhea, mild fever | ORS + Zinc | Correct |
| Skin Fungus | Circular rash, scaling | Topical antifungal | Correct |
| Drug Interaction | Warfarin + Metformin | Interaction warning | Detected |
| Wound Assessment | Deep cut, redness | Cleaning + antibiotics + referral | Correct |

### Safety Features

1. **Confidence Scoring**: Low-confidence diagnoses trigger referral recommendation
2. **Interaction Alerts**: Severe interactions block dispensing with clear warnings
3. **Alternative Suggestions**: When interactions are found, the system suggests safer alternatives
4. **Referral Logic**: Emergency conditions automatically flagged for hospital transfer
5. **Differential Diagnosis**: Alternative conditions listed to prevent anchoring bias
6. **Transparent Reasoning**: Agentic workflow shows step-by-step reasoning trace

### Potential Impact

**Quantitative:**
- CHWs conduct ~500 million patient visits annually
- Even 1% improvement in diagnosis accuracy = 5 million better outcomes
- Drug interaction checking could prevent thousands of adverse events

**Qualitative:**
- Standardizes care quality across CHWs with varying training levels
- Builds CHW confidence through transparent AI reasoning
- Creates documentation trail for patient visits
- Enables supervision and quality improvement

### Limitations & Future Work

**Current Limitations:**
- Requires GPU for reasonable inference speed
- Image analysis limited to common conditions in training data
- Drug database covers essential medicines only (supplemented by DDInter online)

**Future Development:**
- Fine-tune on regional disease patterns
- Expand drug database with local formularies
- Multi-language support for diverse CHW populations
- Integrate with health information systems for continuity of care

---

## Conclusion

HealthPost demonstrates how MedGemma 1.5's medical AI capabilities can be combined into a practical tool that addresses a real gap in global health. By supporting the complete CHW workflow from intake to dispensing — with transparent agentic reasoning — we can improve both the quality and safety of care for the billions of people who depend on community-based healthcare.

---

**Repository:** [GitHub Link]

**Demo Video:** [3-minute demonstration]

**Contact:** [Team Information]

---

*Built for the MedGemma Impact Challenge 2025*
