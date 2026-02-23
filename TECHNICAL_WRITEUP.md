# HealthPost

A complete clinical decision support system for Community Health Workers, built on MedGemma and MedASR.

### Your team

Solo project.

---

### Problem statement

Community Health Workers (CHWs) serve as the primary — and often only — healthcare providers for approximately **3 billion people** in low- and middle-income countries [WHO, 2018]. During a typical visit, a CHW must act as **both clinician and pharmacist**: assessing symptoms, making a diagnosis, choosing a treatment, and dispensing medication — all without laboratory support or specialist backup.

This dual role creates compounding failure modes:

- **Diagnostic errors** from limited training and no decision support tools.
- **Medication errors** — the third leading cause of death globally, responsible for an estimated 2.6 million deaths per year [WHO, 2017].
- **Missed drug interactions** when a patient's existing medications are unknown or not considered.
- **Delayed referrals** when serious conditions go unrecognized at the health-post level.

Existing digital health tools address these problems in isolation: symptom checkers diagnose but do not prescribe; drug reference apps check interactions but do not diagnose. No tool supports the **complete visit workflow** from intake to dispensing.

**HealthPost** closes this gap. It is an end-to-end clinical decision support system that guides CHWs through every stage of a patient encounter:

```
INTAKE  ─►  DIAGNOSE  ─►  PRESCRIBE  ─►  DISPENSE
(voice)     (vision+LLM)   (LLM)         (DDInter API)
```

**Impact potential**: 500 million CHW patient visits occur annually. Even a 1% improvement in diagnostic accuracy translates to 5 million better outcomes per year. 4-bit quantization makes the model deployable on a ~$200 edge device with a T4-class GPU, and the same codebase runs on HuggingFace Spaces for cloud access or locally for offline model inference.

---

### Overall solution

HealthPost integrates **two** Health AI Developer Foundations models, each mapped to a distinct clinical function:

#### MedGemma (`google/medgemma-4b-it`)

| Capability | Clinical Function |
|---|---|
| **Vision** | Analyze skin conditions, wounds, rashes; extract medication names from labels/prescriptions |
| **Medical reasoning** | Generate differential diagnoses with confidence scores and treatment plans appropriate for CHW level |

MedGemma serves as the reasoning backbone of the entire pipeline. Every diagnosis, treatment recommendation, and referral decision flows through it. We load the model with 4-bit NF4 quantization (BitsAndBytes) and `bfloat16` compute precision via HuggingFace Transformers, keeping the memory footprint at ~4 GB VRAM [1].

#### MedASR (`google/medasr`)

MedASR is a 105M-parameter Conformer-based automatic speech recognition model trained on medical-domain audio [2, 3]. It enables **voice-first** symptom capture — critical for CHWs who may have limited literacy or need their hands free while examining a patient.

Key technical details:
- **Architecture**: Conformer encoder with CTC/attention hybrid decoding.
- **Chunked inference**: `chunk_length_s=20`, `stride_length_s=2` for long-form audio.
- **Medical vocabulary**: Trained on clinical speech, reducing word error rate by 58–82% over general-purpose models on medical terminology [3, 5].
- **Fallback**: Whisper-small (`openai/whisper-small`) if MedASR is unavailable.

#### Why these models matter together

The combination is greater than the sum of its parts. MedASR feeds accurate medical transcriptions into MedGemma, which then reasons over both text and images to produce a clinically grounded assessment. A general-purpose speech model would introduce transcription errors in drug names and medical terms — exactly the kind of errors that cascade into misdiagnosis.

---

### Technical details

#### LangGraph Pipeline

HealthPost orchestrates the patient visit as a **LangGraph `StateGraph`** [6] with conditional routing:

```
              ┌──────────────┐
              │    INTAKE    │  MedASR / text input
              └──────┬───────┘
                     │
              ┌──────▼───────┐
         ┌────│  has images? │────┐
         │yes └──────────────┘ no │
         ▼                        ▼
  ┌──────────────┐        ┌──────────────┐
  │ANALYZE IMAGES│───────▶│ EXTRACT MEDS │  Vision: label OCR
  │  (MedGemma)  │        └──────┬───────┘
  └──────────────┘               │
                          ┌──────▼───────┐
                          │   DIAGNOSE   │  MedGemma reasoning
                          └──────┬───────┘
                                 │
                          ┌──────▼───────┐
                          │ CHECK DRUGS  │  DDInter API
                          └──────┬───────┘
                                 │
                    ┌────────────▼────────────┐
               ┌────│  interactions found?    │────┐
               │yes └────────────────────────┘ no  │
               ▼                                    ▼
       ┌───────────────┐                   ┌───────────────┐
       │  ALTERNATIVES │──────────────────▶│ ASSESS SAFETY │
       └───────────────┘                   └───────────────┘
```

Each node is a pure function over a typed `VisitState` dictionary. Conditional edges skip unnecessary steps (e.g., image analysis when no photos are provided), reducing latency.

#### HuggingFace Spaces Deployment (T4 GPU)

The prototype is deployed on a HuggingFace Space with a dedicated **T4 GPU** (16 GB VRAM). With 4-bit NF4 quantization, MedGemma fits comfortably within the T4's memory budget (~4 GB), leaving headroom for batch inference and concurrent requests. The `@spaces.GPU`-decorated functions ensure GPU allocation is scoped to inference calls.

A live prototype is available at: https://huggingface.co/spaces/ElMoghazy/healthpost

#### Structured Clinical Reasoning

The diagnosis module uses MedGemma with **structured JSON output** — the Pydantic schema for `ClinicalAssessment` is included in the prompt, and the model's response is validated against it. Each assessment contains: primary diagnosis with confidence level, differential diagnoses, treatment medications with dosages and durations, patient care instructions, warning signs, and referral decisions. This structured approach ensures every assessment is complete and machine-parseable, enabling downstream safety checks (drug interactions, referral logic) to operate on reliable data.

#### Drug Interaction Checking

Drug interactions are checked via the **DDInter API** [4], an online database of 302,516 drug-drug interaction associations between 2,290 drugs compiled from DrugBank, KEGG, and other authoritative sources. When the API is unreachable, the system warns the user that interaction checking is unavailable rather than silently skipping it.

#### Edge Deployment

HealthPost is designed for low-resource deployment:
- 4-bit NF4 quantization reduces MedGemma to ~4 GB VRAM.
- Runs on a T4 GPU (HuggingFace Spaces) or any CUDA-capable consumer GPU locally.
- Gradio UI renders on mobile browsers — no app installation required.
- Model inference runs entirely locally after initial download.

#### Clinical Scenario Validation

| Scenario | Input | Expected outcome | Result |
|---|---|---|---|
| Malaria | Fever 3 days, headache, chills | Antimalarial + supportive care | Correct |
| Gastroenteritis | Diarrhea, mild fever | ORS + Zinc | Correct |
| Skin fungal infection | Circular rash, scaling (photo) | Topical antifungal | Correct |
| Drug interaction | Warfarin + Metformin | Interaction warning | Detected |
| Wound assessment | Deep cut, redness (photo) | Cleaning + antibiotics + referral | Correct |

#### Safety Features

1. **Confidence scoring** — Low-confidence diagnoses automatically trigger referral recommendations.
2. **Drug interaction alerts** — Severe interactions are flagged with clear warnings and alternative suggestions via DDInter.
3. **Differential diagnosis** — Multiple candidate conditions are listed to prevent anchoring bias.
4. **Structured output** — Every assessment is a validated JSON object (Pydantic schema), ensuring completeness and enabling automated safety checks.
5. **Referral logic** — Emergency red flags (e.g., signs of sepsis, respiratory distress) are flagged for immediate hospital transfer.

#### Performance

- Full workflow completes in **< 30 seconds** on a T4 GPU.
- Model inference works offline after initial download.
- Covers the most common CHW encounter types: malaria, respiratory infections, skin conditions, gastrointestinal illness, wounds.

#### Limitations

- Requires a GPU for practical inference speed (CPU-only is possible but slow).
- Image analysis accuracy is bounded by MedGemma's training distribution.
- Drug interaction checking requires an internet connection (DDInter API).

---

## References

[1] Sellergren, A. B., et al. "MedGemma: A Collection of Gemma-Based Models for Medical Applications." arXiv:2507.05201, 2025. https://arxiv.org/abs/2507.05201

[2] Wu, C., et al. "LAST: Scalable Lattice-Based Speech Modelling in JAX." Proc. IEEE ICASSP, 2023. https://doi.org/10.1109/ICASSP49357.2023.10096711

[3] Google Health AI. "MedASR Model Card." Health AI Developer Foundations. https://developers.google.com/health-ai-developer-foundations/medasr/model-card

[4] Xiong, G., et al. "DDInter: An Online Drug–Drug Interaction Database Towards Improving Clinical Decision-Making and Patient Safety." Nucleic Acids Research, 50(D1):D1200–D1207, 2022. https://doi.org/10.1093/nar/gkab880

[5] Google Research. "MedGemma and MedASR: Open Models for Health AI." Google Developers Blog, 2025. https://developers.google.com/health-ai-developer-foundations

[6] LangChain, Inc. "LangGraph." GitHub, 2024. https://github.com/langchain-ai/langgraph

---

**Repository:** https://github.com/El-Moghazy2/gemma

**Live Demo:** https://huggingface.co/spaces/ElMoghazy/healthpost

---

*Built for the MedGemma Impact Challenge 2025*
