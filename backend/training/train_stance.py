"""
train_stance.py - Fine-tune DeBERTa-v3 on FEVER + VitaminC

Datasets:
  - FEVER: 185k Wikipedia claim/evidence pairs (SUPPORTS/REFUTES/NOT ENOUGH INFO)
  - VitaminC: counterfactual augmentation, teaches sensitivity to small factual changes

Why VitaminC matters:
  "aspirin reduces fever" vs "aspirin does not reduce fever"
  Standard NLI models often miss these small negations. VitaminC specifically
  trains robustness to counterfactual edits — the key failure mode in fact-checking.

Label mapping (consistent with DeBERTa NLI convention):
  REFUTES / contradiction   → 0
  NOT ENOUGH INFO / neutral → 1
  SUPPORTS / entailment     → 2
"""

import os
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME   = "microsoft/deberta-v3-base"
OUTPUT_DIR   = "./checkpoints/stance_deberta"
MAX_LENGTH   = 512
BATCH_SIZE   = 16          # per-device; effective = 16 * grad_accum
GRAD_ACCUM   = 2           # effective batch size = 32
EPOCHS       = int(os.getenv("TRAIN_EPOCHS", 3))
LR           = 2e-5
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
SEED         = 42

# Caps dataset size for a smoke test on a machine with no GPU (validates the
# script end to end in minutes instead of hours); unset for the real run.
MAX_TRAIN_SAMPLES = os.getenv("MAX_TRAIN_SAMPLES")
MAX_VAL_SAMPLES = os.getenv("MAX_VAL_SAMPLES")

# Label space
LABEL2ID = {"REFUTES": 0, "NOT ENOUGH INFO": 1, "SUPPORTS": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_fever():
    """
    FEVER: Fact Extraction and VERification

    The original "fever" repo on the Hub ships a loading script, which datasets
    >=3.0 refuses to execute. pietrolesci/nli_fever is the same FEVER data
    re-hosted as Parquet with gold evidence already joined in, which is also
    why it needs no reassembly here. Its own field names are inverted from NLI
    convention -- "premise" there is the claim and "hypothesis" is the
    evidence -- so they are swapped below to match this file's convention
    (premise = evidence, hypothesis = claim).
    """
    logger.info("Loading FEVER dataset...")
    dataset = load_dataset("pietrolesci/nli_fever")

    def process_fever(example):
        label_str = example["fever_gold_label"]
        if label_str not in LABEL2ID:
            return None
        return {
            "premise":    example["hypothesis"],
            "hypothesis": example["premise"],
            "label":      LABEL2ID[label_str],
        }

    def filter_and_map(split):
        processed = []
        for ex in dataset[split]:
            result = process_fever(ex)
            if result and result["premise"]:
                processed.append(result)
        return Dataset.from_list(processed)

    train = filter_and_map("train")
    val   = filter_and_map("dev")
    logger.info(f"FEVER — train: {len(train)}, val: {len(val)}")
    return train, val


def load_vitaminc():
    """
    VitaminC: counterfactual fact verification dataset
    Critically improves robustness to small factual edits — the exact failure
    mode standard NLI models exhibit on real-world fact-checking.
    Format is already (evidence, claim, label), directly usable.
    """
    logger.info("Loading VitaminC dataset...")
    dataset = load_dataset("tals/vitaminc")

    # VitaminC label strings: "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
    def process_vitaminc(example):
        label_str = example["label"]
        if label_str not in LABEL2ID:
            return None
        return {
            "premise":    example["evidence"],
            "hypothesis": example["claim"],
            "label":      LABEL2ID[label_str]
        }

    def filter_and_map(split):
        processed = []
        for ex in dataset[split]:
            result = process_vitaminc(ex)
            if result and result["premise"]:
                processed.append(result)
        return Dataset.from_list(processed)

    train = filter_and_map("train")
    val   = filter_and_map("validation")
    logger.info(f"VitaminC — train: {len(train)}, val: {len(val)}")
    return train, val


# ── Tokenization ─────────────────────────────────────────────────────────────

def build_tokenize_fn(tokenizer):
    """
    NLI input format: [CLS] premise [SEP] hypothesis [SEP]
    We pass (premise, hypothesis) as sentence pair — tokenizer handles the
    segment IDs and separator tokens automatically.
    """
    def tokenize(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,   # dynamic padding in DataCollator
        )
    return tokenize


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    macro_f1 = f1_score(labels, predictions, average="macro")
    per_class = f1_score(labels, predictions, average=None)
    
    return {
        "macro_f1":       macro_f1,
        "f1_refutes":     per_class[0],
        "f1_nei":         per_class[1],
        "f1_supports":    per_class[2],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)

    # Logged up front, unconditionally: a driver/CUDA-build mismatch makes
    # torch.cuda.is_available() silently return False, which would run the
    # whole job on CPU and blow through the walltime limit with barely any
    # progress. This is cheaper to check here than in a separate interactive
    # debug session.
    cuda_ok = torch.cuda.is_available()
    logger.info(
        "torch=%s cuda_build=%s cuda_available=%s",
        torch.__version__, torch.version.cuda, cuda_ok,
    )
    if cuda_ok:
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning(
            "No CUDA GPU visible to torch -- training will run on CPU and is "
            "very unlikely to finish within the job's time limit."
        )

    # Load and combine datasets
    fever_train, fever_val   = load_fever()
    vitaminc_train, vitaminc_val = load_vitaminc()

    # Combine: FEVER is the primary dataset, VitaminC adds counterfactual robustness
    train_dataset = concatenate_datasets([fever_train, vitaminc_train]).shuffle(seed=SEED)
    val_dataset   = concatenate_datasets([fever_val, vitaminc_val]).shuffle(seed=SEED)

    if MAX_TRAIN_SAMPLES:
        train_dataset = train_dataset.select(range(min(int(MAX_TRAIN_SAMPLES), len(train_dataset))))
    if MAX_VAL_SAMPLES:
        val_dataset = val_dataset.select(range(min(int(MAX_VAL_SAMPLES), len(val_dataset))))

    logger.info(f"Combined train: {len(train_dataset)}, val: {len(val_dataset)}")

    # Tokenizer + model
    # dtype is explicit because from_pretrained's default ("auto") picks up
    # this checkpoint's stored float16 weights, which produces NaN logits
    # from the first forward pass on both CPU and MPS -- fp16 mixed precision
    # is handled separately below via TrainingArguments(fp16=...).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        dtype=torch.float32,
        # microsoft/deberta-v3-base only ships pytorch_model.bin on its main
        # revision; newer transformers refuses to torch.load that format
        # below torch 2.6 (CVE-2025-32434) regardless of whether the load
        # would actually be safe. Forcing safetensors makes it fetch the
        # community PR that adds a safetensors copy instead of raising, and
        # avoids torch.load's real security exposure entirely as a bonus.
        use_safetensors=True,
    )

    tokenize_fn = build_tokenize_fn(tokenizer)
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["premise", "hypothesis"])
    val_dataset   = val_dataset.map(tokenize_fn,   batched=True, remove_columns=["premise", "hypothesis"])

    # warmup_ratio was removed (not just deprecated) in newer transformers
    # versions than this was first written against -- compute the equivalent
    # step count by hand instead of depending on either name existing.
    steps_per_epoch = -(-len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM))  # ceil div
    warmup_steps = int(WARMUP_RATIO * steps_per_epoch * EPOCHS)

    # Training arguments
    # fp16 enabled for GPU; automatically ignored if no CUDA
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="steps",
        eval_steps=int(os.getenv("EVAL_STEPS", 1000)),
        save_strategy="steps",
        save_steps=int(os.getenv("EVAL_STEPS", 1000)),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        report_to="none",    # swap to "wandb" when W&B is configured
        seed=SEED,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")

    # Final evaluation with full classification report
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    print("\n" + classification_report(labels, preds, target_names=["REFUTES", "NOT ENOUGH INFO", "SUPPORTS"]))


if __name__ == "__main__":
    main()
