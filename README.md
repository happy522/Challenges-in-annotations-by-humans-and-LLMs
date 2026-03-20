# Challenges in Annotations by Humans and LLMs

This repository contains code, data, and experimental artifacts for the paper:

> **Challenges in annotations by humans and LLMs: A case study of evaluative language** 
> M. Imamovic, A. Knierim, K. Pitroda, E. Lapshinova‑Koltunski

The project compares annotations produced by linguists in training, an expert linguist, and large language models (LLMs) on complex evaluative language phenomena in English TED talks, using Appraisal theory as the linguistic framework. 

***

## Overview

We study evaluative language (stance/opinion) in spoken popular science discourse (TED talks), focusing on the **Attitude** subsystem of Appraisal theory with three main classes:
- Affect (feelings and emotions)  
- Judgement (evaluations of people’s behaviour)  
- Appreciation (evaluations of things, processes, or states)

The repository supports:

- Human annotation experiments on TED talk transcripts  
- LLM‑based Appraisal classification with different prompting strategies  
- Quantitative evaluation (agreement measures, F1‑scores) and qualitative error analysis
***

## Research Questions

The underlying study addresses: 

1. Which Attitude categories are challenging for human annotators, and why?  
2. Do LLMs meet expectations for complex annotation task resolution in Appraisal theory?  
3. Do humans and LLMs struggle with the same issues when annotating evaluative language?

The best model setup in the paper reaches an F1‑score of 0.77 for Attitude classification on the TED talk corpus. 
***

## Repository Structure

```text
.
├── data/
│   ├── raw/                 # Original TED talk transcripts
│   ├── processed/           # Preprocessed sentence-level data
│   ├── annotations_humans/  # Annotations by students and expert linguist
│   └── annotations_llms/    # Model-generated labels
├── prompts/
│   ├── zero_shot.txt
│   ├── few_shot.txt
│   └── chain_of_thought.txt
├── src/
│   ├── preprocessing.py
│   ├── run_llm_annotation.py
│   ├── evaluation.py
│   └── analysis/
│       ├── agreement_analysis.py
│       └── error_analysis.py
├── results/
│   ├── metrics/
│   └── figures/
├── LICENSE
└── README.md
```
***

## Setup

1. Clone the repository: [github](https://github.com/happy522)
   ```bash
   git clone https://github.com/happy522/Challenges-in-annotations-by-humans-and-LLMs.git
   cd Challenges-in-annotations-by-humans-and-LLMs
   ```

2. Create and activate an environment (example with `conda`):  
   ```bash
   conda create -n appraisal-llms python=3.10
   conda activate appraisal-llms
   ```

3. Install dependencies (replace with your actual file):  
   ```bash
   pip install -r requirements.txt
   ```

***

## Data

- **Source**: English TED talk transcripts from eleven domains (spoken popular science / specialised communication).
- **Unit of analysis**: Sentence-level, with one or more evaluative items per sentence; explicit and implicit evaluations are included.
- **Labels**: Attitude classes (Affect, Judgement, Appreciation), plus polarity (positive/negative).

***

## Human Annotation

The repository can include:
 
- Annotation files from:
  - Linguists in training (students)  
  - An expert linguist (gold standard)
    
The study highlights typical challenges:

- Deciding the span of evaluative expressions  
- Distinguishing Judgement vs. Appreciation  
- Handling implicit vs. explicit Attitude and context dependence 
Agreement metrics such as Cohen’s kappa and F1 are used to quantify inter‑annotator reliability. 

***

## LLM Annotation

We experiment with several LLMs and prompting strategies for automatic Appraisal classification: 
- Prompt families: zero‑shot, few‑shot, and more structured prompts  
- Models: three different LLMs (qwen3_30b_a3b_instruct_2507, llama_3_3_70b_instruct, mistral_large_instruct)

The pipeline typically consists of:

1. Preparing sentence‑level input files  
2. Generating predictions with each prompt/model  
3. Comparing LLM labels against the expert gold standard  
4. Computing overall scores and per‑class performance

The best‑performing setup achieves an F1‑score of 0.77 and aligns most closely with the expert linguist, while students show lower agreement.

***


***

## Results and Findings

Key high‑level findings from the paper: 

- LLMs can reach or surpass the agreement levels of linguists in training on Appraisal Attitude classification.  
- The expert linguist remains the most consistent reference, but LLMs approximate that performance with careful prompt design and fine‑tuning.  
- Certain Attitude categories and implicit evaluations remain challenging for both humans and models.  
- Clear, detailed guidelines and span‑level annotation are crucial to improve reliability for complex evaluative phenomena.

***

