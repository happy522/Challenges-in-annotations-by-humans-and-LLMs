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

# =========================
# CONFIG
# =========================

MODEL_NAME = "unsloth/Qwen3-30B-A3B-Instruct-2507"
INPUT_CSV = "gold_standard_with_context.csv"
OUTPUT_DIR = "./qwen_eval_binary_lora"

MAX_LENGTH = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 2e-4

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(INPUT_CSV)

df = df.rename(columns={
    "Sentence": "target_text",
    "Prev_Sentence": "prev_text",
    "Next_Sentence": "next_text",
    "Evaluative": "label",
})

df["label"] = df["label"].fillna(0).astype(int)

dataset = Dataset.from_pandas(
    df[["prev_text", "target_text", "next_text", "label"]],
    preserve_index=False,
)

# =========================
# TRAIN / TEST SPLIT (60 / 40)
# =========================

split_ds = dataset.train_test_split(test_size=0.4, seed=42)

train_ds = split_ds["train"]
test_ds = split_ds["test"]
train_ds.to_pandas().to_csv("training.csv")
# Save test set for later inference
test_ds.to_pandas().to_csv("testing.csv", index=False)
print(f"Saved testing.csv with {len(test_ds)} examples")

# =========================
# PROMPT (UNCHANGED)
# =========================

def build_prompt(example):
    label_text = "Evaluative" if example["label"] == 1 else "Non-evaluative"

    prompt = f"""You are an annotation assistant for Appraisal Theory (Martin & White, 2005).

    Task:
    You are an annotation assistant for Appraisal Theory (Martin & White, 2005).

    Task:
    Decide whether the TARGET text contains any evaluative content.

    You are given:
    - PREVIOUS: text from the row above (may be empty)
    - TARGET: the current text to label
    - NEXT: text from the row below (may be empty)

    Use PREVIOUS and NEXT only as context (e.g., to resolve pronouns or topics).
    The label must refer ONLY to the TARGET text.

    Rule “Evaluative” vs “Non-Evaluative” as follows:

    - Evaluative
      -> If the TARGET text expresses an emotion, a judgment of people/behaviour,
        or an aesthetic/quality/value evaluation of things, performances or phenomena.

    - Non-Evaluative
      -> If the TARGET text is purely factual, descriptive, or neutral, with no clear
        emotional stance, no judgment of people/behaviour, and no quality/value
        evaluation of things or events.

    Examples (illustrative only):
    - “The room is beautiful.” -> evaluative: aesthetic evaluation of a thing.
    - “She was very unfair to her team.” -> evaluative: judgment of behaviour.
    - “The concert made me really happy.” -> evaluative: emotional reaction.
    - “The meeting starts at 3 pm.” -> non-evaluative: purely factual.

    Important constraints:
    - Do NOT invent or assume information that is not explicitly present in the
      TARGET text or necessary to interpret it in context. No hallucinations.
    - If you are unsure, choose the more conservative option and explain briefly.

    Context:

    PREVIOUS:
    \"\"\"{example['prev_text']}\"\"\"

    TARGET:
    \"\"\"{example['target_text']}\"\"\"

    NEXT:
    \"\"\"{example['next_text']}\"\"\""

    Answer with exactly one word: "Evaluative" or "Non-evaluative"


    Answer:
    """

    return {
        "text": prompt + " " + label_text,
        "label": example["label"],
    }

train_ds = train_ds.map(build_prompt)

# =========================
# TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    input_ids = list(tokens["input_ids"])
    attention_mask = list(tokens["attention_mask"])

    labels = input_ids.copy()

    label_word = "Evaluative" if example["label"] == 1 else "Non-evaluative"
    prompt_part = example["text"].rsplit(label_word, 1)[0]
    prompt_len = len(
        tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
    )

    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized_train_ds = train_ds.map(
    tokenize,
    remove_columns=train_ds.column_names,
    batched=False,
)

# =========================
# QLoRA MODEL (SINGLE GPU)
# =========================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": "cuda:0"},
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# TRAINER
# =========================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=3,
    learning_rate=LR,
    bf16=True,
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    ddp_find_unused_parameters=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    data_collator=data_collator,
)

trainer.train()

# =========================
# SAVE LORA ADAPTER
# =========================

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA adapter saved to:", OUTPUT_DIR)
