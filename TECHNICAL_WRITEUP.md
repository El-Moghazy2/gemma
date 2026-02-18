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

By integrating MedGemma's medical AI capabilities with an offline drug database, HealthPost provides:
1. Visual analysis of skin conditions and wounds
2. AI-powered diagnosis based on symptoms
3. Treatment recommendations appropriate for health post level
4. Drug interaction checking before dispensing

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
│  │  Symptoms   │─▶│  MedGemma   │─▶│  MedGemma   │     │
│  │  (text)     │  │  Vision     │  │  Reasoning  │     │
│  │             │  │             │  │             │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                            │            │
│                                            ▼            │
│                                    ┌─────────────┐     │
│                                    │  DISPENSE   │     │
│                                    │             │     │
│                                    │  Drug DB    │     │
│                                    │  (SQLite)   │     │
│                                    │             │     │
│                                    └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### HAI-DEF Models Used

| Model | Capability | Application |
|-------|------------|-------------|
| **MedGemma 4B** | Medical Vision | Analyze skin conditions, wounds, rashes |
| **MedGemma 4B** | Medical Reasoning | Generate diagnosis and treatment plans |

### Key Components

**1. Medical Image Analysis**

MedGemma Vision analyzes uploaded photos with condition-specific prompts:
- Skin conditions: Identifies rashes, lesions, infections
- Wounds: Assesses type, infection signs, healing stage
- Provides severity assessment and recommended actions

**2. Diagnosis Engine**

MedGemma Text processes symptoms and visual findings to generate:
- Primary diagnosis with confidence level
- Differential diagnoses to consider
- Treatment recommendations appropriate for CHW level
- Referral guidance when hospital care is needed

**3. Drug Safety Module**

Offline SQLite database containing:
- WHO Essential Medicines List (~300 drugs)
- Known drug-drug interactions with severity ratings
- Contraindications and dosing guidance

Before dispensing, the system checks for interactions between:
- Patient's current medications
- Newly recommended medications

**4. Edge Deployment**

Designed for offline operation:
- 4-bit quantization reduces model size by 75%
- SQLite database requires no internet connection
- Gradio UI works on mobile browsers

### Technical Specifications

```
Model: MedGemma 4B (quantized)
Memory: ~4GB VRAM with 4-bit quantization
Inference: <10 seconds per diagnosis on T4 GPU
Database: ~5MB SQLite (300 drugs, 50 interactions)
Interface: Gradio web UI (mobile-compatible)
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
| Drug Interaction | Metformin + Alcohol | Severe warning | Detected |

### Safety Features

1. **Confidence Scoring**: Low-confidence diagnoses trigger referral recommendation
2. **Interaction Alerts**: Severe interactions block dispensing with clear warnings
3. **Referral Logic**: Emergency conditions automatically flagged for hospital transfer
4. **Differential Diagnosis**: Alternative conditions listed to prevent anchoring bias

### Potential Impact

**Quantitative:**
- CHWs conduct ~500 million patient visits annually
- Even 1% improvement in diagnosis accuracy = 5 million better outcomes
- Drug interaction checking could prevent thousands of adverse events

**Qualitative:**
- Standardizes care quality across CHWs with varying training levels
- Builds CHW confidence in decision-making
- Creates documentation trail for patient visits
- Enables supervision and quality improvement

### Limitations & Future Work

**Current Limitations:**
- Requires GPU for reasonable inference speed
- Image analysis limited to common conditions in training data
- Drug database covers essential medicines only

**Future Development:**
- Fine-tune on regional disease patterns
- Expand drug database with local formularies
- Add voice input via MedASR for low-literacy users
- Integrate with health information systems for continuity of care

---

## Conclusion

HealthPost demonstrates how MedGemma's medical AI capabilities can be combined into a practical tool that addresses a real gap in global health. By supporting the complete CHW workflow from intake to dispensing, we can improve both the quality and safety of care for the billions of people who depend on community-based healthcare.

---

**Repository:** [GitHub Link]

**Demo Video:** [3-minute demonstration]

**Contact:** [Team Information]

---

*Built for the MedGemma Impact Challenge 2025*
