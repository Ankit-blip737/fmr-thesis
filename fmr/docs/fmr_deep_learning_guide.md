# FMR Project — Complete Deep-Dive Learning Guide

> This guide teaches you every concept, every term, and every stage of your thesis project from absolute fundamentals to the final dashboard. Read it sequentially — each module builds on the previous one.

---

## Module 1: The Problem — Why Does This Project Exist?

### 1.1 What is happening in hospitals right now?

Doctors are overwhelmed. A single radiologist may need to read 100+ X-rays, CT scans, and MRIs per day. AI companies are building tools that can "read" these images and answer clinical questions automatically — like "Is there a fracture in this X-ray?" or "What organ is affected?"

### 1.2 What is the danger?

The AI sometimes **lies confidently**. It might say "Yes, there is a fracture" even when the image shows a perfectly healthy bone. Worse, it might give the right answer — but for the **wrong reason**. For example, it might answer "Yes, fracture" because it learned that 70% of X-ray questions have "Yes" as the answer, not because it actually saw a crack in the bone.

### 1.3 What did you build?

You built a **safety auditor** — a system that watches the AI think, measures whether it is actually looking at the image, and **pulls the emergency brake** (says "I don't know, ask a human doctor") if the AI is not trustworthy on a specific case.

### 1.4 Why is this important?

No one else has built this exact combination before. Individual pieces exist (VLMs exist, conformal prediction exists), but your specific pipeline that:
- Measures grounding decay step-by-step
- Fuses multiple faithfulness signals
- Gates the output with a distribution-free safety guarantee

...is a novel contribution to the field.

---

## Module 2: The Building Blocks — Key Terminology

### 2.1 What is a "Model"?

A **model** is a mathematical function (with billions of numbers called "weights" or "parameters") that has been trained on massive amounts of data. After training, you give it an input (like an image), and it produces an output (like text). Think of it as a very sophisticated pattern-matching machine.

### 2.2 What is a VLM (Vision-Language Model)?

A **Vision-Language Model** is a special type of AI model that can understand **both** images and text simultaneously. 

- A regular language model (like ChatGPT) can only read and write text.
- A VLM can **look at an image** AND read text, and then generate a text response about that image.

**How it works internally:**
1. The image is broken into small patches (like a grid of tiny squares).
2. Each patch is converted into a numerical "embedding" (a list of numbers that represents what that patch contains).
3. These image embeddings are fed into the language model alongside the text question.
4. The language model generates an answer token-by-token (one word at a time).

### 2.3 What is Medical VQA (Visual Question Answering)?

**Medical VQA** is the specific task of:
- **Input:** A medical image (X-ray, CT scan, MRI, pathology slide) + a clinical question ("Is there a tumor?")
- **Output:** A text answer ("Yes, there is a mass in the left lung")

Your project uses three real Medical VQA datasets:
- **VQA-RAD** — Radiology images (X-rays, CTs, MRIs) with clinical questions
- **PathVQA** — Pathology slides (microscope images of tissue) with questions
- **SLAKE** — A bilingual (English/Chinese) medical VQA dataset with multiple modalities

### 2.4 What are the two models in your project?

| Model | Full Name | Role | Why chosen? |
|-------|-----------|------|-------------|
| **MedVLM-R1** | `JZPeterPan/MedVLM-R1` | The "Reasoning" model | A medical VLM specifically fine-tuned to think step-by-step (chain-of-thought) before answering. It is 4.42 GB. |
| **Qwen2.5-VL-3B** | `Qwen/Qwen2.5-VL-3B-Instruct` | The "Non-Reasoning" model | A general-purpose VLM that answers quickly and directly, without showing its reasoning steps. Used as a baseline comparison. |

**Why two models?** To prove that chain-of-thought reasoning doesn't automatically make a model more faithful. Your thesis shows that MedVLM-R1 *reasons more*, but that reasoning can still drift away from the image (grounding decay).

### 2.5 What is "Hallucination"?

**Hallucination** = when the AI generates text that sounds correct and confident, but is factually wrong or not supported by the input data.

Example:
- **Image:** A normal, healthy chest X-ray
- **Question:** "Is there pneumonia?"
- **Hallucinated answer:** "Yes, bilateral infiltrates suggest pneumonia" ← This is a lie. The AI made it up.

### 2.6 What is "Grounding"?

**Grounding** = the degree to which the AI's answer is actually based on (grounded in) the input image.

- **Well-grounded:** The AI says "fracture" because it detected a crack in the bone pixels.
- **Poorly grounded:** The AI says "fracture" because the word "fracture" appeared frequently in its training data whenever it saw an X-ray question.

### 2.7 What is "Grounding Decay"?

**Grounding Decay** is the central hypothesis of your thesis. It means:

> As the VLM generates more reasoning steps (longer chain-of-thought), it progressively "forgets" the original image and starts relying more on its language patterns.

Imagine a doctor looking at an X-ray. At first, they focus on the image. But as they write a longer and longer report, they start copying phrases from memory instead of looking back at the X-ray. That's grounding decay.

Your project **measures** this decay step-by-step and **proves** it happens.

---

## Module 3: The Datasets — What Data Did You Use?

### 3.1 Mock (Synthetic) Dataset

**What:** A completely fake, computer-generated dataset where YOU control every variable.

**Why it exists:** To prove your math works in a controlled environment. Because you generated the data yourself, you know the exact "ground truth" — you know exactly where the disease is, exactly what the correct answer is, and exactly how faithful the reasoning should be. This lets you verify that your formulas produce the right numbers.

**Key property:** The mock dataset has **bounding boxes** (exact X, Y coordinates of the "disease" region in the image). This is critical for Signal B (measuring if the model is looking at the right spot).

### 3.2 Real Datasets

| Dataset | Images | Modalities | Answer Type | Has Bounding Boxes? |
|---------|--------|------------|-------------|---------------------|
| **VQA-RAD** | ~300 | X-ray, CT, MRI | Open + Closed (yes/no) | ❌ No |
| **PathVQA** | ~5000 | Pathology slides | Open + Closed | ❌ No |
| **SLAKE** | ~640 | X-ray, CT, MRI | Open + Closed | ❌ No |

**Key limitation:** None of the real datasets have bounding boxes. This is why the "Measurement" (AUROC) and "Robustness" tabs are empty on the Real data — you can't measure spatial grounding without knowing where the disease actually is.

### 3.3 What is "Closed" vs "Open" question type?

- **Closed (Multiple Choice):** "Is there a fracture? (A) Yes (B) No" → The model picks from a fixed set of options.
- **Open (Free-form):** "Describe the abnormality in this image" → The model generates any text it wants.

Closed questions are easier to evaluate (just check if the letter matches). Open questions require a "judge" to decide if the answer is semantically correct.

---

## Module 4: The Pipeline — Stage by Stage

Your project runs in **5 sequential stages**. Each stage feeds its output into the next one.

### Stage 1: Baselines

**File:** `run_real.py` → baselines stage

**What happens:**
1. Take each image from the dataset.
2. Feed it to **both** models (MedVLM-R1 and Qwen2.5-VL-3B).
3. Each model generates an answer.
4. Compare each answer to the ground truth.
5. Record the accuracy of each model.

**Why:** This establishes the "raw performance" of each model. It answers: "How accurate is each model on this dataset, ignoring everything else?"

**Output:** `baselines.json` — a file containing the accuracy numbers for both models.

### Stage 2: Blind Test

**File:** `run_real.py` → blind_test stage

**What happens:**
1. Take each image from the dataset.
2. Create a **completely blank (black) image** of the same size.
3. Feed the **same question** to the model twice:
   - Once with the **real image** → get an answer
   - Once with the **blank image** → get an answer
4. Compare the two answers.

**Why:** This is the "smoking gun" test for grounding. If the model gives the **same answer** whether it sees the real X-ray or a blank black square, then the model is clearly NOT looking at the image. It's just guessing based on the text of the question.

**Key metric — Blind Gap:**
$$\text{Blind Gap} = \text{Accuracy}_{\text{real image}} - \text{Accuracy}_{\text{blank image}}$$

- **Blind Gap > 0:** Good! The model performs better with the real image, meaning it IS using the image.
- **Blind Gap ≈ 0:** Bad! The model performs equally well with a blank image, meaning it's ignoring the image entirely.
- **Blind Gap < 0:** Very bad! The model actually performs WORSE with the real image (the image is confusing it).

**Output:** `blind_test.json` — accuracy on real vs blank images, and the blind gap.

### Stage 3: FMR (Faithful Medical Reasoning) Score

This is the **heart of your thesis**. It computes a single number (the "Faithfulness Score" or FS) that measures how trustworthy a specific answer is.

The FS is computed by **fusing three independent signals:**

---

#### Signal A: Counterfactual (Image Reliance)

**File:** `fmr/src/fmr/faithfulness/counterfactual.py`

**The idea:** If the model is truly relying on the image, then changing the image should change the answer.

**What happens:**
1. Show the model the **real image** + question → get Answer₁
2. Show the model a **blank image** + question → get Answer₂  
3. Show the model a **mismatched image** (a random different medical image from the dataset) + question → get Answer₃
4. Compare: Did the answers change when the image changed?

**The math:**
$$\text{Signal A} = 1 - \frac{\text{similarity}(\text{Answer}_1, \text{Answer}_2) + \text{similarity}(\text{Answer}_1, \text{Answer}_3)}{2}$$

- **Signal A ≈ 1.0:** The model's answer changed dramatically when the image changed → it IS relying on the image. Good!
- **Signal A ≈ 0.0:** The model gave the same answer regardless of which image it saw → it's ignoring the image. Bad!

---

#### Signal B: Attention/Grounding (Spatial Focus)

**File:** `fmr/src/fmr/faithfulness/attention.py`

**The idea:** Even if the model changes its answer when the image changes (Signal A is high), is it looking at the **right part** of the image?

**What happens:**
1. Extract the model's "attention maps" — these are heatmaps showing which pixels the model focused on while generating each reasoning step.
2. Compare these attention regions to the **ground-truth bounding box** (the area where the disease actually is).
3. Measure the overlap using IoU (Intersection over Union).

**The math:**
$$\text{Signal B} = \text{IoU}(\text{model's attention region}, \text{ground-truth disease region})$$

- **Signal B ≈ 1.0:** The model is looking directly at the disease. 
- **Signal B ≈ 0.0:** The model is looking at the wrong part of the image.

**Critical limitation:** Signal B requires bounding boxes. Since the real datasets (VQA-RAD, PathVQA, SLAKE) don't have bounding boxes, Signal B can only be computed on the **Mock dataset**. On real data, Signal B defaults to 0.5 (neutral). This is why the "Measurement" tab is empty on Real data.

---

#### Signal C: Consistency (Answer Stability)

**File:** `fmr/src/fmr/faithfulness/consistency.py`

**The idea:** If you ask the same question 5 times (with slight randomness in the model's sampling), a faithful model should give the same answer every time. An unfaithful model will give different answers each time because it's guessing.

**What happens:**
1. Ask the model the same (image, question) pair **5 times** (this is `n_consistency=5`).
2. Each time, the model uses a slightly different random seed (controlled by `temperature=0.7`), which introduces small variations in its token sampling.
3. Count how many of the 5 answers agree with each other.

**The math:**
$$\text{Signal C} = \frac{\text{number of times the most common answer appeared}}{5}$$

- **Signal C = 1.0:** All 5 answers were identical → the model is very confident and consistent. 
- **Signal C = 0.2:** Each of the 5 answers was different → the model is randomly guessing. 

**Why 5 passes?** This is why your Colab took 3.5 hours. For 300 images × 5 passes = 1,500 forward passes through a 4.42 GB model. Each pass generates a full chain-of-thought (sometimes hundreds of tokens). This is the computational bottleneck.

**What is "temperature"?**
Temperature controls how "random" the model's word choices are:
- **Temperature = 0.0:** The model always picks the most likely next word. Completely deterministic.
- **Temperature = 0.7:** The model sometimes picks slightly less likely words, introducing variety.
- **Temperature = 1.0+:** The model becomes very random, almost incoherent.

You use 0.7 because it's high enough to reveal inconsistency (if the model is guessing, different runs will diverge), but low enough that a truly confident model will still give the same answer.

---

#### Fusion: Combining Signals into the Faithfulness Score (FS)

**File:** `fmr/src/fmr/faithfulness/score.py`

**The idea:** Each signal alone is incomplete. Signal A tells you IF the model uses the image, Signal B tells you WHERE it looks, and Signal C tells you HOW confident it is. You need all three together.

**The math:**
$$\text{FS} = w_A \cdot \text{Signal A} + w_B \cdot \text{Signal B} + w_C \cdot \text{Signal C}$$

Where $w_A$, $w_B$, $w_C$ are learned weights (called `DEFAULT_WEIGHTS` in the code) that determine how much each signal contributes.

- **FS close to 1.0:** The model's answer is highly faithful — it used the image, looked at the right spot, and gave a consistent answer.
- **FS close to 0.0:** The model's answer is untrustworthy — it ignored the image, looked at the wrong spot, or gave inconsistent answers.

---

### Stage 4: Conformal Abstention (The Safety Gate)

**File:** `fmr/src/fmr/abstention/conformal.py`

This is the **mathematical safety guarantee** — the most academically impressive part of your thesis.

#### 4.1 What is the problem?

You now have an FS score for every case. But what threshold do you use to decide "safe" vs "unsafe"? If you set it too high, the AI will refuse to answer anything (useless). If you set it too low, the AI will answer cases it shouldn't (dangerous).

#### 4.2 What is Conformal Prediction?

**Conformal Prediction** is a branch of statistics that provides **distribution-free guarantees**. 

"Distribution-free" means: the guarantee works **regardless of what the data looks like**. You don't need to assume the data follows a bell curve, or any specific shape. It works on ANY data distribution. This is incredibly powerful for medicine, where data can be weird and unpredictable.

#### 4.3 How does the Conformal Gate work?

1. **Split the data** into two halves:
   - **Calibration set** (e.g., 75 cases): Used to find the threshold.
   - **Test set** (e.g., 75 cases): Used to verify the threshold works.

2. **Set your risk tolerance ($\alpha$):**
   - You choose $\alpha = 0.15$, meaning: "I want the AI to be wrong on at most 15% of the cases it chooses to answer."

3. **Find the threshold ($\tau$):**
   - Sort all calibration cases by their FS score.
   - Find the FS cutoff where, if you only answer cases with $\text{FS} \geq \tau$, the error rate among answered cases is $\leq \alpha$.

4. **The guarantee:**
   - Conformal theory proves that this threshold, calibrated on the calibration set, will **also** control the error rate on **future, unseen data** — with probability $\geq 1 - \delta$ (where $\delta = 0.05$).

#### 4.4 What is "Abstention"?

When the AI encounters a new case:
- If $\text{FS} \geq \tau$: The AI **answers** (it is confident enough).
- If $\text{FS} < \tau$: The AI **abstains** (it says "I don't know, please ask a human doctor").

This is exactly what the green "ANSWER" and red "ABSTAIN" badges mean on your dashboard.

#### 4.5 Key metrics on the dashboard

| Metric | What it means |
|--------|---------------|
| **Coverage** | The percentage of cases the AI chose to answer (didn't abstain on). Higher = more useful. |
| **Retained Error** | The error rate among the cases the AI chose to answer. Must be ≤ α. |
| **AURC (Area Under Risk-Coverage Curve)** | A single number summarizing the tradeoff between coverage and error. Lower = better. |

#### 4.6 Why is this better than just using confidence?

Traditional AI systems use the model's own "confidence score" to decide when to abstain. But models are often **confidently wrong** — they give high confidence to hallucinated answers. Your FS score is external and independent: it measures faithfulness from the outside, not from the model's self-assessment.

---

### Stage 5: Correction (Optional — MedGemma Verifier)

**File:** `fmr/src/fmr/correction/pipeline.py`

**The idea:** For cases where the AI abstained, can a second, independent model "verify" or "correct" the answer?

**What happens:**
1. Take cases where MedVLM-R1 abstained.
2. Send them to a second model (**MedGemma-4B**) for a second opinion.
3. If MedGemma agrees with MedVLM-R1's original answer, maybe we can "rescue" that case and let it through.

**Status:** This is partially implemented. The Qwen model works; MedGemma had issues with image sensitivity (it gave the same answer regardless of the image), so it's flagged as "Future Work."

---

## Module 5: The Dashboard — What Each Tab Shows

### 5.1 Overview Tab

A high-level summary card showing:
- **Baselines accuracy** for both models
- **Blind gap** (does the model use the image?)
- **Coverage** and **retained error** at the current α
- **Replication verdict** — an honest, automated assessment of whether the grounding decay hypothesis is supported by this dataset

### 5.2 Diagnosis Tab

Two main visualizations:
- **Blind Test Bar Chart:** Compares accuracy on real images vs blank images for both models. The gap between the bars is the "blind gap."
- **Per-Step Grounding Curve (Signal B):** Shows how the model's attention to the correct image region changes at each reasoning step. If this curve goes down, it proves grounding decay. (Only available on Mock data because real datasets lack bounding boxes.)

### 5.3 Measurement Tab

- **AUROC (Area Under ROC Curve):** Measures how well the FS score separates "correct" cases from "incorrect" cases. A perfect separator would have AUROC = 1.0. Random guessing = 0.5.
- **Risk-Coverage Curve:** As you lower the FS threshold (letting more cases through), how does the error rate change? The ideal curve stays flat and low.

### 5.4 Robustness Tab

Shows **ablation studies** — what happens when you deliberately break things:
- **Noise ablation:** Add random noise to the image and see if FS drops.
- **Crop ablation:** Crop the image and see if FS drops.
- **Blur ablation:** Blur the image and see if FS drops.
- **Shuffle ablation:** Shuffle the image patches and see if FS drops.

These prove your FS score is robust and sensitive to real image degradation. Only available on Mock data (needs bounding boxes).

### 5.5 Case Explorer Tab

A searchable, filterable table of every individual case in the dataset. For each case, you can see:
- The question asked
- The model's answer
- Whether the answer was correct
- The FS score
- The ANSWER/ABSTAIN decision
- The individual signal values (A, B, C)

### 5.6 Source Picker (Top of page)

A dropdown that lets you switch between:
- `Mock (Synthetic)` — the controlled experiment
- `Real — VQA-RAD` — real radiology images
- `Real — PathVQA` — real pathology images
- `Real — SLAKE` — real multi-modality images

---

## Module 6: The Code Architecture — How Files Connect

```
fmr/
├── src/fmr/                          ← The core Python library
│   ├── types.py                      ← Sample dataclass (image + question + answer)
│   ├── models/
│   │   ├── hf_vlm.py                ← HFVLM class (loads MedVLM-R1 / Qwen on GPU)
│   │   ├── mock_vlm.py              ← MockVLM (fake model for testing without GPU)
│   │   └── second_vlm.py            ← SecondVLM (MedGemma verifier wrapper)
│   ├── faithfulness/
│   │   ├── counterfactual.py         ← Signal A (image reliance)
│   │   ├── attention.py              ← Signal B (spatial grounding)
│   │   ├── consistency.py            ← Signal C (answer stability)
│   │   └── score.py                  ← Fuses A+B+C → FS
│   ├── abstention/
│   │   └── conformal.py              ← The safety gate (calibrate_threshold)
│   ├── correction/
│   │   └── pipeline.py               ← Stage 5 correction with MedGemma
│   └── data/
│       ├── loaders.py                ← Loads VQA-RAD, PathVQA, SLAKE, OmniMedVQA
│       └── synthetic.py              ← Generates the Mock dataset
│
├── scripts/
│   ├── run_real.py                   ← The main pipeline (runs Stages 1-4 on real data)
│   ├── run_fmr.py                    ← Runs the FMR score computation
│   ├── make_figures.py               ← Generates thesis figures
│   └── make_dashboard.py             ← Bundles everything into data.js
│
├── dashboard/
│   ├── index.html                    ← The web page structure
│   ├── style.css                     ← All the visual styling (dark theme, gradients)
│   ├── app.js                        ← All the interactive logic (charts, filters, tabs)
│   └── data.js                       ← The pre-computed results (generated by make_dashboard.py)
│
├── notebooks/
│   ├── colab_real_pipeline.ipynb     ← The main Colab notebook (runs on free GPU)
│   ├── colab_stage4_correction_real.ipynb  ← MedGemma correction notebook
│   └── colab_faithfulness_lora.ipynb ← LoRA fine-tuning ablation notebook
│
├── api.py                            ← FastAPI backend for live demo
└── Dockerfile.backend                ← Docker container for self-hosting
```

---

## Module 7: Key Mathematical Concepts

### 7.1 What is AUROC?

**Area Under the Receiver Operating Characteristic Curve.**

Imagine you have a pile of 100 medical cases. 50 are correct and 50 are incorrect. Your FS score ranks them from most faithful to least faithful.

- **Perfect FS:** All 50 correct cases have higher FS than all 50 incorrect cases. AUROC = 1.0.
- **Useless FS:** Correct and incorrect cases are randomly mixed. AUROC = 0.5.
- **Your goal:** AUROC as close to 1.0 as possible.

### 7.2 What is AURC?

**Area Under the Risk-Coverage Curve.**

As you slide the FS threshold from high to low:
- **Coverage increases** (you answer more cases).
- **Risk (error rate) may increase** (you start answering harder cases).

AURC measures the total area under this curve. **Lower AURC = better** (you can cover more cases while keeping risk low).

### 7.3 What is IoU (Intersection over Union)?

Used in Signal B to measure spatial overlap between two rectangles (the model's attention region and the ground-truth disease region).

$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

- **IoU = 1.0:** Perfect overlap (the model is looking exactly at the disease).
- **IoU = 0.0:** No overlap (the model is looking at a completely different part of the image).

### 7.4 What is α (alpha)?

The **maximum error rate you are willing to tolerate** among the cases the AI answers. In your project, α = 0.15 means: "Of all the cases the AI chooses to answer, at most 15% can be wrong."

### 7.5 What is δ (delta)?

The **probability that the guarantee fails**. δ = 0.05 means: "There is a 5% chance that the error rate exceeds α on new data." Combined with α, this gives you a (1-δ) = 95% confidence that the error guarantee holds.

### 7.6 What is τ (tau)?

The **FS threshold** computed by the conformal gate. Any case with FS ≥ τ gets answered; any case with FS < τ gets abstained.

---

## Module 8: The End-to-End Flow (Putting It All Together)

Here is exactly what happens when you run the pipeline on a dataset like SLAKE:

```
Step 1: Load 150 SLAKE images from HuggingFace
            ↓
Step 2: Run MedVLM-R1 and Qwen on each image → baselines.json
            ↓
Step 3: Run MedVLM-R1 on each image + blank image → blind_test.json
            ↓
Step 4: For each image, compute:
        • Signal A (show real/blank/mismatch image, measure answer change)
        • Signal B (extract attention, compare to bounding box — Mock only)
        • Signal C (ask 5 times, measure consistency)
        • Fuse → FS score
            ↓
Step 5: Split 150 cases into 75 calibration + 75 test
            ↓
Step 6: On the 75 calibration cases, find threshold τ
         such that answered cases have ≤ 15% error
            ↓
Step 7: On the 75 test cases, apply τ:
         FS ≥ τ → ANSWER (green badge)
         FS < τ → ABSTAIN (red badge)
            ↓
Step 8: Bundle everything into data.js
            ↓
Step 9: Push to GitHub → Dashboard auto-updates
```

---

## Module 9: Why "Mock" and "Real" Exist Side-by-Side

| Aspect | Mock (Synthetic) | Real (VQA-RAD, PathVQA, SLAKE) |
|--------|-------------------|--------------------------------|
| **Purpose** | Prove the math works | Prove it works on real hospitals |
| **Images** | Computer-generated | Real X-rays, CTs, MRIs |
| **Bounding Boxes** | ✅ Yes (you placed them) | ❌ No (datasets don't have them) |
| **Signal B works?** | ✅ Yes | ❌ No (defaults to 0.5) |
| **Robustness tab?** | ✅ Full ablations | ❌ Shows Mock data with badge |
| **AUROC?** | ✅ Computable | ❌ Needs bounding boxes |
| **Blind Test?** | ✅ Yes | ✅ Yes |
| **Case Explorer?** | ✅ Yes | ✅ Yes |

The Mock dataset is your **proof of concept**. The Real datasets are your **proof of applicability**. Together, they make a complete thesis.

---

## Module 10: The Thesis Argument (How to Present This)

Your thesis tells this story:

1. **"Medical VLMs hallucinate."** → You prove this with the Blind Test (some models perform equally well on blank images).

2. **"Chain-of-thought reasoning doesn't fix it."** → You prove this by showing that MedVLM-R1, despite reasoning step-by-step, still suffers from grounding decay (the per-step curve drops on Mock data).

3. **"We can detect it."** → You prove this by showing that your 3-signal FMR score (A+B+C) has high AUROC — it successfully separates faithful answers from unfaithful ones.

4. **"We can prevent it."** → You prove this by showing that the Conformal Abstention gate, using the FMR score as its input, achieves the promised error rate (≤ α) with a distribution-free mathematical guarantee.

5. **"It works on real medical data."** → You prove this by running the pipeline on VQA-RAD, PathVQA, and SLAKE — three real, peer-reviewed medical datasets.

This is a complete, rigorous, publishable research contribution.
