# Faithful Medical Reasoning (FMR)
## A Safety Framework for Medical Vision-Language Models

**Ankit Kumar**
B.Tech Thesis | Department of Computer Science

---

# 1. The Clinical Problem: Confident but Wrong

### The Danger of Medical AI Hallucination
Vision-Language Models (VLMs) are increasingly used to interpret X-rays and pathology slides. However, a critical reliability gap remains:
- **Hallucination:** Models often generate confident diagnoses that are entirely incorrect.
- **The "Clever Hans" Effect:** Models can arrive at the *correct* diagnosis by memorizing text patterns in the question, without actually looking at the medical image.
- **Grounding Decay:** As models use "Chain-of-Thought" reasoning, they drift away from the visual evidence and rely solely on language patterns.

> **The Challenge:** How can a doctor trust an AI if we don't know whether it actually looked at the patient's scan?

---

# 2. Our Innovation: The FMR Framework

### A First-of-its-Kind Safety Auditor
We built **Faithful Medical Reasoning (FMR)** — a comprehensive system that sits on top of any medical AI to audit its thought process. 

Rather than trying to blindly increase accuracy, FMR:
1. **Measures** how faithfully the model uses the image (The Faithfulness Score).
2. **Detects** when the model's reasoning drifts away from the visual evidence.
3. **Gates** the output, issuing a mathematically guaranteed **ANSWER** (safe) or **ABSTAIN** (defer to human) decision.

---

# 3. System Architecture & Pipeline

### End-to-End Evaluation Infrastructure
We engineered a modular, 5-stage pipeline that processes raw medical datasets through complex AI inference and outputs interactive web visualizations.

- **Stage 1: Baselines:** Establishes the raw accuracy of reasoning vs. non-reasoning models.
- **Stage 2: Blind Test:** Evaluates the model on blank images to detect pure hallucination.
- **Stage 3: FMR Score:** Computes our novel 3-signal faithfulness metric.
- **Stage 4: Safety Gate:** Applies Conformal Prediction to find the safety threshold.
- **Stage 5: Live API:** A FastAPI backend allowing real-time, dynamic inference.

---

# 4. Exposing Hallucination: The Blind Test

### Proving When the Model Ignores the Image
To prove that medical AI hallucinates, we developed the **Blind Test**. We ask the AI the same clinical question twice:
1. Once with the **real X-ray**.
2. Once with a **completely blank (black) image**.

**The Blind Gap Metric:** 
If the model gives the exact same correct answer on the blank image, it proves the model is guessing from text priors, not diagnosing from visual evidence. Our pipeline successfully exposes this critical flaw in modern VLMs.

---

# 5. The Faithfulness Score (FS)

### Fusing Three Independent Signals
Our framework calculates a definitive Faithfulness Score (FS) by measuring three distinct behaviors:

- **Signal A (Image Reliance):** We swap the image with counterfactuals (blank, mismatched). A faithful model *must* change its answer.
- **Signal B (Spatial Grounding):** We extract the AI's internal attention heatmaps and calculate the IoU (Intersection over Union) with the actual disease location.
- **Signal C (Answer Consistency):** We query the model 5 times with slight temperature variations. A grounded model answers consistently; a guessing model fluctuates.

$$FS = w_A \cdot A + w_B \cdot B + w_C \cdot C$$

---

# 6. Proving Grounding Decay

### More Reasoning ≠ More Faithful
A major finding of our project is the visualization of **Grounding Decay**. 

By tracking Signal B (Spatial Grounding) across the model's "Chain-of-Thought", we proved that:
- **Step 1:** The model starts by focusing correctly on the disease region.
- **Steps 2–3:** The model's attention begins to drift.
- **Steps 4+:** The model essentially stops looking at the image and writes purely from its language memory.

Our system successfully captures and graphs this decay, proving that forcing an AI to "think more" can actually increase hallucination.

---

# 7. The Conformal Safety Gate

### Distribution-Free Mathematical Guarantees
How do we know when the Faithfulness Score is high enough to trust? We implemented a **Conformal Prediction Gate**.

- **Calibration:** We split the data and define a strict risk tolerance (e.g., α = 0.15, meaning ≤15% error).
- **Thresholding:** The math finds a precise threshold (τ) that guarantees this error rate.
- **The Guarantee:** With 95% confidence, the system will maintain this safety level on unseen future data.

> **Result:** The system outputs a definitive **ANSWER** (Safe) or **ABSTAIN** (Unsafe), protecting patients from unverified AI guesses.

---

# 8. Experimental Setup & Engineering

### Rigorous Testing Across Multiple Modalities
We successfully engineered the pipeline to run across a diverse array of medical datasets, requiring complex GPU orchestration.

- **Mock (Synthetic):** A highly controlled dataset we built to mathematically prove the FMR signals work perfectly when ground-truth is known.
- **VQA-RAD:** Real clinical radiology (X-rays, CTs, MRIs).
- **PathVQA:** Real pathology microscope slides.
- **SLAKE:** Multi-modal, bilingual clinical imaging.

**Infrastructure Built:** The entire pipeline was executed on NVIDIA T4 GPUs via Google Colab, processing thousands of complex chain-of-thought inference passes.

---

# 9. The FMR Interactive Dashboard

### Bringing Data to Life
We built a production-ready, interactive web dashboard (deployed via Vercel) to visualize the massive amounts of data generated by the pipeline.

- **Dynamic Visualizations:** AUROC charts, Risk-Coverage curves, and Per-Step Grounding decay graphs.
- **Case Explorer:** A searchable interface to review every single medical image, the AI's reasoning chain, and its final FMR score.
- **Robustness Analytics:** Real-time data on how the model performs under image noise, blurring, and cropping.

---

# 10. The Live Clinical API

### Real-Time AI Auditing
Beyond static analysis, we engineered a complete **Live Backend Architecture** for future clinical deployment.

- **FastAPI Server:** We built a Python web server that hosts the MedVLM-R1 model.
- **Dynamic Inference:** Doctors can upload a new, unseen X-ray through the dashboard. The image is sent to the GPU via an Ngrok tunnel.
- **Real-Time Safety Check:** The backend runs all 5 consistency passes live and returns the final FMR score and Conformal Gate decision in seconds.

---

# 11. Conclusion & Impact

### What We Accomplished
We designed, engineered, and proved a complete end-to-end safety auditor for medical AI. 

By combining complex Vision-Language Model inference with rigorous Conformal Prediction math, we proved that **we can mathematically detect when an AI is hallucinating**. 

We have provided the medical AI industry with a critical tool: a system that doesn't just ask the AI for an answer, but verifies whether the AI actually looked at the patient before speaking.
