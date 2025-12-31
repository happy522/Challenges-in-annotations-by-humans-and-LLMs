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
    accuracy_score,
    precision_recall_fscore_support,
)

# =========================================================
# CONFIG
# =========================================================
BASE_MODEL = "unsloth/Qwen3-30B-A3B-Instruct-2507"
LORA_DIR = "./qwen_eval_binary_lora"
TEST_CSV = "testing_multi.csv"

MAX_NEW_TOKENS = 5
DEVICE = "cuda"

LABEL_MAP = {
    "evaluative": 1,
    "non-evaluative": 0,
}

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(TEST_CSV)

df = df.rename(columns={
    "Sentence": "target_text",
    "Prev_Sentence": "prev_text",
    "Next_Sentence": "next_text",
})

df["label"] = df["label"].astype(int)
y_true = df["label"].values

print("Evaluation samples:", len(df))

# =========================================================
# PROMPT (MUST MATCH TRAINING)
# =========================================================
PROMPT = """
You are an annotation assistant for Appraisal Theory (Martin & White, 2005).

Task:
Decide whether the TARGET text contains any evaluative content.

You are given:
- PREVIOUS (context only)
- TARGET (this is what you label)
- NEXT (context only)

Rules:
- Evaluative: emotion, judgement, or value/quality evaluation
- Non-evaluative: purely factual or neutral

Important constraints:
- Base your decision ONLY on TARGET (use context only if needed)
- Do NOT invent information

PREVIOUS:
\"\"\"{prev_text}\"\"\"

TARGET:
\"\"\"{target_text}\"\"\"

NEXT:
\"\"\"{next_text}\"\"\"

Answer with exactly one word:
Evaluative or Non-evaluative

Answer:
""".strip()

def build_prompt(row):
    return PROMPT.format(
        prev_text=row["prev_text"] or "",
        target_text=row["target_text"] or "",
        next_text=row["next_text"] or "",
    )

# =========================================================
# LOAD MODEL (4-bit + LoRA)
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
# GENERATION + PARSING (STRICT)
# =========================================================
def generate_binary_label(prompt):
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
    answer = decoded.split("Answer:")[-1].strip().lower()

    # normalize
    answer = answer.replace(".", "").replace("–", "-")
    print(answer)
    if "non-evaluative" in answer:
        return 0
    if "evaluative" in answer:
        return 1

    # fallback (conservative)
    return 0

# =========================================================
# RUN INFERENCE
# =========================================================
y_pred = []

print("Running binary inference...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    prompt = build_prompt(row)
    pred = generate_binary_label(prompt)
    y_pred.append(pred)

y_pred = np.array(y_pred)

# =========================================================
# METRICS
# =========================================================
print("\n=== BINARY CLASSIFICATION REPORT ===")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Non-evaluative", "Evaluative"],
    zero_division=0
))

acc = accuracy_score(y_true, y_pred)
p, r, f, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {p:.4f}")
print(f"Recall   : {r:.4f}")
print(f"F1       : {f:.4f}")

# =========================================================
# SAVE
# =========================================================
df_out = df.copy()
df_out["binary_pred"] = y_pred

out_path = os.path.join(LORA_DIR, "binary_evaluation_predictions.csv")
df_out.to_csv(out_path, index=False)
print("\nSaved predictions to:", out_path)

# =========================================================
# CLEANUP
# =========================================================
del model, base_model
gc.collect()
torch.cuda.empty_cache()