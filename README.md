# Challenges in Annotations by Humans and LLMs

This repository contains the code, experimental setup, and supporting materials for the paper:

**"Challenges in annotations by humans and LLMs: A case study of evaluative language"**

---

## 📌 Overview

This project investigates how **human annotators** and **large language models (LLMs)** perform on a **complex linguistic annotation task**: identifying and classifying *evaluative language* using **Appraisal Theory**.

We compare:

* Linguists in training
* A trained linguist (gold standard)
* Multiple LLMs (Qwen, Llama, Mistral)

The study focuses on:

* Detecting whether a sentence is **evaluative** or not.
* Classifying it into **Appraisal categories**:

  * Affect (emotions)
  * Judgement (ethics/behavior)
  * Appreciation (aesthetics/quality)

---

## 🎯 Research Questions

1. Which Appraisal categories are most challenging for human annotators?
2. Can LLMs handle complex annotation tasks effectively?
3. Do humans and LLMs struggle with the same issues?

---

## 📊 Key Results

* **Binary classification (evaluative vs non-evaluative)**:

  * LLMs achieve solid performance (~F1 ≈ 0.68–0.70)
* **Multiclass classification (Appraisal categories)**:

  * Performance drops significantly (macro F1 ≈ 0.30–0.52)
* **LLMs vs Humans**:

  * LLMs outperform **non-expert annotators**
  * Performance is comparable to a **trained linguist**
* **Hardest category**:

  * **Judgement** (for both humans and models)

---

## 📂 Repository Structure

```
├── classified data/
│   ├── raw/                 # Original TED talk transcripts
│   ├── student data/        # annotations by students
│   └── llm data/            # Model-generated labels
├── experiments/
│   ├── fine tuning/      # scripts for fine tuning the model
│   ├── results script/   # scripts for results generation
│   ├── Binary_Multiclass_classification.ipynb    # classification script for LLMs with best prompt
│   └── Data_Generation_for_LLMs.ipynb            # data generation for LLMs
├── prompt analysis/
│   ├── prompts/          # 3 versions of prompt
│   └── analysis/         # analysis of prompts
├── results/
│   ├── llm/              # LLM results
│   └── student/          # student restuls
└── README.md
```

---

## 🧠 Methodology

### 1. Human Annotation

* Dataset: TED Talk transcripts (EmotionalizTED corpus)
* Annotation level: **sentence-level**
* Annotators:

  * 24 linguists in training
  * 1 senior researcher (gold standard)
* Metrics:

  * Cohen’s Kappa
  * Krippendorff’s Alpha

* Two-step pipeline:

  1. Binary classification (evaluative vs non-evaluative)
  2. Multiclass classification (Appraisal categories)

### 2. LLM Annotation

* Models used:

  * Qwen3_30b_a3b_instruct_2507
  * Llama_3_3_70b_instruct
  * Mistral_large_instruct
  * 
* Two-step pipeline:

  1. Binary classification (evaluative vs non-evaluative)
  2. Multiclass classification (Appraisal categories)

### 3. Prompt Engineering

Three prompt strategies were tested:

* Chain-of-thought few-shot
* Zero-shot
* Structured few-shot (best performing)

### 4. Fine-tuning

* Method: **QLoRA (4-bit quantization)**
* Result:

  * Improved **binary detection**
  * Worse **fine-grained classification**

---

## ⚠️ Key Challenges

### Human Annotation

* Low inter-annotator agreement (high subjectivity)
* Confusion between:

  * Judgement vs Appreciation
* Difficulty with:

  * Implicit meanings
  * Target of evaluation
  * Long sentences

### LLMs

* Bias toward predicting **Affect**
* Low recall for **Judgement**
* Sensitive to prompt design
* Struggle with multi-label classification

---

## 🔍 Insights

* Complex linguistic theories (like Appraisal) are **hard for both humans and machines**
* LLMs are **useful assistants**, not replacements
* Annotation quality strongly depends on:

  * Clear guidelines
  * Task design
  * Level of granularity (sentence vs token/span)

---

## 🚀 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/happy522/Challenges-in-annotations-by-humans-and-LLMs.git
cd Challenges-in-annotations-by-humans-and-LLMs
```

### 2. Run annotation experiments

* Use prompt files in `/prompts`
* Run scripts in `/models` or `/evaluation`

### 3. Evaluate results

* Metrics:

  * Precision
  * Recall
  * F1-score
  * Krippendorff’s Alpha

---

## 📈 Future Work

* Shift from **sentence-level → span/token-level annotation**
* Improve handling of:

  * Implicit vs explicit evaluation
* Better prompt strategies for:

  * Multi-label classification
* Address class imbalance

---

## 🤝 Contributions

Contributions are welcome!
Feel free to open issues or submit pull requests.
---

## 📬 Contact

For questions or collaboration:

* Open an issue on GitHub
* Contact the authors via institutional emails

---
