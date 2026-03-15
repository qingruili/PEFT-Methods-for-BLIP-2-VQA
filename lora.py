"""
LoRA fine-tuning of BLIP-2 (OPT-2.7B) on VQA v2.
Trains and saves adapter weights to checkpoints/lora/.
Run evaluate.py separately to test the saved model.
"""

import json
import time
from pathlib import Path

import torch
from peft import PeftModel

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

TRAIN_SIZE  = 30000
BATCH_SIZE  = 2       # safe limit for 8GB VRAM with BLIP-2 8-bit
NUM_EPOCHS  = 3
LR          = 1e-4
GRADIENT_ACCUMULATION_STEPS = 4   # effective batch = BATCH_SIZE * GRAD_ACCUM = 8

LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

CHECKPOINT_DIR = Path("checkpoints/lora")
# ──────────────────────────────────────────────────────────────────────────────


def find_resume_epoch():
    """Return the latest completed epoch number to resume from, or 0 if starting fresh."""
    for epoch in range(NUM_EPOCHS, 0, -1):
        epoch_dir = CHECKPOINT_DIR / f"epoch_{epoch}"
        if (epoch_dir / "adapter_config.json").exists():
            return epoch
    return 0


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


def load_lora_from_checkpoint(model, epoch):
    """Load LoRA weights from a saved epoch checkpoint."""
    epoch_dir = CHECKPOINT_DIR / f"epoch_{epoch}"
    model = PeftModel.from_pretrained(model, str(epoch_dir), is_trainable=True)
    return model


def make_batch_inputs(batch, processor):
    images     = [s["image"]    for s in batch]
    eos = processor.tokenizer.eos_token or ""
    full_texts = [f"Question: {s['question']} Answer: {s['answer']}{eos}" for s in batch]
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
    # Use attention_mask (not pad_token_id) to find padding.
    # OPT has eos_token_id == pad_token_id (both = 2), so masking by
    # pad_token_id would accidentally mask the EOS token too — the model
    # would never learn to stop generating.
    labels[inputs["attention_mask"] == 0] = -100

    inputs["labels"] = labels
    return inputs


def train_one_epoch(model, processor, loader, optimizer):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        inputs = make_batch_inputs(batch, processor)
        loss   = model(**inputs).loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

    # Handle any remaining steps not divisible by GRADIENT_ACCUMULATION_STEPS
    if (len(loader) % GRADIENT_ACCUMULATION_STEPS) != 0:
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def _build_meta(epoch_losses):
    return {
        "method": "lora",
        "config": {
            "train_size": TRAIN_SIZE, "num_epochs": NUM_EPOCHS,
            "lr": LR, "batch_size": BATCH_SIZE,
            "grad_accum": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES,
        },
        "epoch_losses": epoch_losses,
    }


def save_epoch_checkpoint(model, processor, epoch, epoch_losses):
    """Save checkpoint after each epoch for crash recovery."""
    epoch_dir = CHECKPOINT_DIR / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(epoch_dir))
    processor.save_pretrained(str(epoch_dir))
    meta = _build_meta(epoch_losses)
    meta["last_epoch"] = epoch
    with open(epoch_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Epoch {epoch} checkpoint → {epoch_dir}/")


def save_checkpoint(model, processor, epoch_losses):
    """Save final checkpoint to the main directory (used by evaluate.py)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(CHECKPOINT_DIR))
    processor.save_pretrained(str(CHECKPOINT_DIR))
    meta = _build_meta(epoch_losses)
    with open(CHECKPOINT_DIR / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Final checkpoint → {CHECKPOINT_DIR}/")


def main():
    print("=" * 55)
    print("  LoRA Fine-tuning — BLIP-2 on VQA v2")
    print("=" * 55)
    print(f"  train={TRAIN_SIZE}  epochs={NUM_EPOCHS}  lr={LR}  batch={BATCH_SIZE}")
    print(f"  grad_accum={GRADIENT_ACCUMULATION_STEPS}  effective_batch={BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  r={LORA_R}  alpha={LORA_ALPHA}  target={LORA_TARGET_MODULES}")

    # ── Check for existing epoch checkpoints to resume from ──────────────────
    resume_epoch = find_resume_epoch()
    if resume_epoch > 0:
        print(f"\n  [Resume] Found checkpoint at epoch {resume_epoch} — resuming from epoch {resume_epoch + 1}")
    else:
        print("\n  [Fresh] No checkpoint found — starting from epoch 1")

    print("\n[1] Loading model...")
    processor = load_processor()
    model     = load_model()
    model     = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    if resume_epoch > 0:
        model = load_lora_from_checkpoint(model, resume_epoch)
        print(f"  Loaded LoRA weights from epoch_{resume_epoch}/")
    else:
        model = apply_lora(model)

    model.print_trainable_parameters()

    print("\n[2] Loading train dataset...")
    train_data   = VQAv2Dataset(get_fixed_train_subset()[:TRAIN_SIZE], split="train")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: x)
    print(f"  {len(train_data)} samples")

    print("\n[3] Training...")
    optimizer    = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    epoch_losses = []

    for epoch in range(resume_epoch + 1, NUM_EPOCHS + 1):
        t0      = time.perf_counter()
        loss    = train_one_epoch(model, processor, train_loader, optimizer)
        elapsed = time.perf_counter() - t0
        epoch_losses.append(round(loss, 4))
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  loss={loss:.4f}  time={elapsed:.1f}s")
        save_epoch_checkpoint(model, processor, epoch, epoch_losses)

    if resume_epoch >= NUM_EPOCHS:
        print("\n  All epochs already completed. Nothing to train.")
    else:
        print("\n[4] Saving final checkpoint...")
        save_checkpoint(model, processor, epoch_losses)
    print("Done.")


if __name__ == "__main__":
    main()
