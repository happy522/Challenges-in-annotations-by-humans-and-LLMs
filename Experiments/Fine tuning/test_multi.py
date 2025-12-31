import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

# =========================================================
# CONFIG
# =========================================================
BASE_MODEL = "unsloth/Qwen3-30B-A3B-Instruct-2507"
LORA_DIR = "./qwen_appraisal_multilabel_lora"
TEST_CSV = "./qwen_eval_binary_lora/binary_evaluation_predictions.csv"

MAX_NEW_TOKENS = 20

DEVICE = "cuda"        # or "cuda:0"
device_map = "auto"    # let HF handle it

LABELS = ["affect", "judgement", "appreciation"]

# =========================================================
# LOAD TEST DATA
# =========================================================
df = pd.read_csv(TEST_CSV)

df = df.rename(columns={
    "Sentence": "target_text",
    "Prev_Sentence": "prev_text",
    "Next_Sentence": "next_text",
})

# evaluate only evaluative rows (same as training)
df = df[df["binary_pred"] == 1].reset_index(drop=True)

for c in ["Affect", "Judgement", "Appreciation"]:
    df[c] = df[c].fillna(0).astype(int)
print("Evaluation samples:", len(df))

# =========================================================
# BUILD GOLD LABEL MATRIX
# =========================================================
y_true = df[["Affect", "Judgement", "Appreciation"]].values

# =========================================================
# PROMPT (MUST MATCH TRAINING)
# =========================================================
LABEL_MARKER = "\n### LABELS:\n"

PROMPT = f"""
You are an annotation assistant for Appraisal Theory (Martin & White, 2005).

Task:
Classify the TARGET text using Appraisal categories:
- affect
- judgement
- appreciation

You are given:
- PREVIOUS (context only)
- TARGET (this is what you label)
- NEXT (context only)

Important constraints:
- Base your decision ONLY on the given texts.
- Do NOT invent extra information.

Output format:
Return ONLY the label or labels as a comma-separated list.
Examples: "affect", "judgement", "affect, appreciation"

PREVIOUS:
\"\"\"{{prev_text}}\"\"\"

TARGET:
\"\"\"{{target_text}}\"\"\"

NEXT:
\"\"\"{{next_text}}\"\"\"
{LABEL_MARKER}
""".strip()

def build_prompt(row):
    return PROMPT.format(
        prev_text=row["prev_text"] or "",
        target_text=row["target_text"] or "",
        next_text=row["next_text"] or "",
    )

# =========================================================
# LOAD MODEL (4-bit + LoRA) ON GPU 1
# =========================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": DEVICE},
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()

device = next(model.parameters()).device
print("Model device:", device)

# =========================================================
# GENERATION + PARSING
# =========================================================
def generate_labels(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    pred_text = decoded.split(LABEL_MARKER)[-1].strip().lower()
    return pred_text

import re

LABELS = {"affect", "judgement", "appreciation"}

LABEL_RE = re.compile(
    r"labels\s*:\s*(.+)",
    flags=re.IGNORECASE | re.DOTALL,
)

def parse_labels(text: str):
    print("RAW OUTPUT:\n", text)

    if not text:
        return []

    m = LABEL_RE.search(text)
    if not m:
        return []

    label_text = m.group(1).lower()

    # stop at first newline / explanation
    label_text = label_text.split("\n")[0]

    # remove punctuation
    label_text = re.sub(r"[^\w,\s]", "", label_text)

    parts = [p.strip() for p in label_text.split(",")]

    return [p for p in parts if p in LABELS]

# =========================================================
# RUN INFERENCE
# =========================================================
y_pred = []

print("Running multilabel inference...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    prompt = build_prompt(row)
    pred_text = generate_labels(prompt)
    labels = parse_labels(pred_text)
    y_pred.append([1 if l in labels else 0 for l in LABELS])
    print(y_pred)
y_pred = np.array(y_pred)

# =========================================================
# METRICS
# =========================================================
print("\n=== PER-LABEL REPORT ===")
print(classification_report(
    y_true,
    y_pred,
    target_names=LABELS,
    zero_division=0
))

for avg in ["micro", "macro", "weighted"]:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average=avg, zero_division=0
    )
    print(f"{avg.upper():8s}  Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")

subset_acc = accuracy_score(y_true, y_pred)
print("\nSubset accuracy (exact match):", subset_acc)

# =========================================================
# SAVE PREDICTIONS
# =========================================================
df_out = df.copy()
df_out["pred_affect"] = y_pred[:, 0]
df_out["pred_judgement"] = y_pred[:, 1]
df_out["pred_appreciation"] = y_pred[:, 2]

out_path = os.path.join(LORA_DIR, "multilabel_evaluation_predictions.csv")
df_out.to_csv(out_path, index=False)
print("\nSaved predictions to:", out_path)

# =========================================================
# CLEANUP
# =========================================================
del model, base_model
gc.collect()
torch.cuda.empty_cache()