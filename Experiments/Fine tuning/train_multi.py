
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =========================================================
# CONFIG
# =========================================================
MODEL_NAME = "unsloth/Qwen3-30B-A3B-Instruct-2507"
INPUT_CSV = "training_multi.csv"
OUTPUT_DIR = "./qwen_appraisal_multilabel_lora"

MAX_LENGTH = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 2e-4
EPOCHS = 3
DEVICE = "cuda:0"

LABELS = ["affect", "judgement", "appreciation"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD + CLEAN DATA
# =========================================================
df = pd.read_csv(INPUT_CSV)

df = df.rename(columns={
    "Sentence": "target_text",
    "Prev_Sentence": "prev_text",
    "Next_Sentence": "next_text",
})

# Keep evaluative rows only
df = df[df["label"] == 1].reset_index(drop=True)

# Clean label columns: NaN → 0, cast to int
for col in ["Affect", "Judgement", "Appreciation"]:
    df[col] = df[col].fillna(0).astype(int)

# =========================================================
# BUILD GOLD LABEL STRING (MULTI-LABEL)
# =========================================================
def build_gold_labels(row):
    labels = []
    if row["Affect"] == 1:
        labels.append("affect")
    if row["Judgement"] == 1:
        labels.append("judgement")
    if row["Appreciation"] == 1:
        labels.append("appreciation")
    return ", ".join(labels) if labels else "none"

df["gold_labels_str"] = df.apply(build_gold_labels, axis=1)
print("Training examples:", len(df))

# =========================================================
# DATASET
# =========================================================
dataset = Dataset.from_pandas(
    df[["prev_text", "target_text", "next_text", "gold_labels_str"]],
    preserve_index=False,
)

# =========================================================
# PROMPT
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

Context:

PREVIOUS:
\"\"\"{{prev_text}}\"\"\"

TARGET:
\"\"\"{{target_text}}\"\"\"

NEXT:
\"\"\"{{next_text}}\"\"\"
{LABEL_MARKER}
"""

# =========================================================
# BUILD TRAINING TEXT
# =========================================================
def build_prompt(example):
    prompt = PROMPT.format(
        prev_text=example["prev_text"] or "",
        target_text=example["target_text"] or "",
        next_text=example["next_text"] or "",
    )
    return {"text": prompt + example["gold_labels_str"]}

dataset = dataset.map(build_prompt)

# =========================================================
# TOKENIZER + MASKING
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    input_ids = tokens["input_ids"]
    labels = input_ids.copy()

    split_idx = example["text"].rfind(LABEL_MARKER)
    prompt_text = example["text"][: split_idx + len(LABEL_MARKER)]
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
    }

tokenized_ds = dataset.map(
    tokenize,
    remove_columns=dataset.column_names,
    batched=False,
)

# =========================================================
# MODEL (QLoRA)
# =========================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": DEVICE},
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================================================
# TRAINING
# =========================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete.")
"""
# =========================================================
# INFERENCE — PROBABILITIES (METHOD B)
# =========================================================
@torch.no_grad()
def yes_no_probability(prompt, label):
    q = prompt + f"Is {label} present? Answer yes or no.\nAnswer:"
    yes_ids = tokenizer(" yes", return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    no_ids = tokenizer(" no", return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

    def logprob(cand_ids):
        ids = tokenizer(q, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        lp = 0.0
        for t in cand_ids[0]:
            logits = model(ids).logits[0, -1]
            lp += torch.log_softmax(logits, dim=-1)[t].item()
            ids = torch.cat([ids, t.view(1, 1)], dim=1)
        return lp

    yes_lp = logprob(yes_ids)
    no_lp = logprob(no_ids)
    probs = torch.softmax(torch.tensor([yes_lp, no_lp]), dim=0)
    return float(probs[0])  # P(yes)

"""
