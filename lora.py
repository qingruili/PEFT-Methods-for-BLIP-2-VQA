"""
LoRA fine-tuning of BLIP-2 (OPT-2.7B) on VQA v2.
Trains and saves adapter weights to checkpoints/lora/.
Run evaluate.py separately to test the saved model.
"""

import json
import time
from pathlib import Path

import torch

if not torch.cuda.is_available():
    print("WARNING: CUDA not found — running on CPU (will be very slow).")
else:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from load_model import load_model, load_processor, DEVICE
from load_dataset import VQAv2Dataset, get_fixed_train_subset


# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_SIZE  = 100
BATCH_SIZE  = 2
NUM_EPOCHS  = 3
LR          = 1e-4

LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

CHECKPOINT_DIR = Path("checkpoints/lora")
# ──────────────────────────────────────────────────────────────────────────────


def apply_lora(model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def make_batch_inputs(batch, processor):
    images     = [s["image"]    for s in batch]
    full_texts = [f"Question: {s['question']} Answer: {s['answer']}" for s in batch]
    prompts    = [f"Question: {s['question']} Answer:" for s in batch]

    inputs = processor(
        images=images,
        text=full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    prompt_enc  = processor.tokenizer(prompts, padding=False, truncation=True, max_length=512)
    prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]

    labels = inputs["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100

    inputs["labels"] = labels
    return inputs


def train_one_epoch(model, processor, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        inputs  = make_batch_inputs(batch, processor)
        loss    = model(**inputs).loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def save_checkpoint(model, processor, epoch_losses):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(CHECKPOINT_DIR))
    processor.save_pretrained(str(CHECKPOINT_DIR))
    # Save training metadata alongside the weights
    meta = {
        "method": "lora",
        "config": {
            "train_size": TRAIN_SIZE, "num_epochs": NUM_EPOCHS,
            "lr": LR, "batch_size": BATCH_SIZE,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES,
        },
        "epoch_losses": epoch_losses,
    }
    with open(CHECKPOINT_DIR / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved → {CHECKPOINT_DIR}/")


def main():
    print("=" * 55)
    print("  LoRA Fine-tuning — BLIP-2 on VQA v2")
    print("=" * 55)
    print(f"  train={TRAIN_SIZE}  epochs={NUM_EPOCHS}  lr={LR}  batch={BATCH_SIZE}")
    print(f"  r={LORA_R}  alpha={LORA_ALPHA}  target={LORA_TARGET_MODULES}")

    print("\n[1] Loading model...")
    processor = load_processor()
    model     = load_model()
    model     = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model     = apply_lora(model)
    model.print_trainable_parameters()

    print("\n[2] Loading train dataset...")
    train_data   = VQAv2Dataset(get_fixed_train_subset()[:TRAIN_SIZE], split="train")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: x)
    print(f"  {len(train_data)} samples")

    print("\n[3] Training...")
    optimizer    = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    epoch_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0   = time.perf_counter()
        loss = train_one_epoch(model, processor, train_loader, optimizer)
        elapsed = time.perf_counter() - t0
        epoch_losses.append(round(loss, 4))
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  loss={loss:.4f}  time={elapsed:.1f}s")

    print("\n[4] Saving...")
    save_checkpoint(model, processor, epoch_losses)
    print("Done.")


if __name__ == "__main__":
    main()
