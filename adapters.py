"""
Bottleneck Adapter fine-tuning of BLIP-2 (OPT-2.7B) on VQA v2.
Injects small bottleneck adapter modules after every OPT decoder layer.
All base model weights are frozen; only the adapter parameters are trained.

  x → LayerNorm → down_proj → GELU → up_proj → + x  (residual)

Saves adapter weights to checkpoints/adapters/.
Run evaluate.py separately to test the saved model.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

if not torch.cuda.is_available():
    print("WARNING: CUDA not found — running on CPU (will be very slow).")
else:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import prepare_model_for_kbit_training

from load_model import load_model, load_processor, DEVICE
from load_dataset import VQAv2Dataset, get_fixed_train_subset


# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_SIZE      = 100
BATCH_SIZE      = 2
NUM_EPOCHS      = 3
LR              = 1e-3    # adapters use a higher lr than LoRA
BOTTLENECK_SIZE = 64

CHECKPOINT_DIR  = Path("checkpoints/adapters")
# ──────────────────────────────────────────────────────────────────────────────


class BottleneckAdapter(nn.Module):
    """
    Residual bottleneck adapter. Near-zero init ensures training starts
    from the pretrained model's behavior (up_proj weights initialised to 0).
    """
    def __init__(self, hidden_size: int, bottleneck_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_size, bias=True)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck_size, hidden_size, bias=True)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


def inject_adapters(model):
    hidden_size    = model.language_model.config.hidden_size
    decoder_layers = model.language_model.model.decoder.layers
    adapters       = nn.ModuleList()
    handles        = []

    def make_hook(adapter):
        def hook(module, inp, output):
            hs      = output[0]
            adapted = adapter(hs.float()).to(hs.dtype)
            return (adapted,) + output[1:]
        return hook

    for layer in decoder_layers:
        adapter = BottleneckAdapter(hidden_size, BOTTLENECK_SIZE).to(DEVICE)
        adapters.append(adapter)
        handles.append(layer.register_forward_hook(make_hook(adapter)))

    # Freeze base model, only adapter params are trainable
    for param in model.parameters():
        param.requires_grad = False
    for param in adapters.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in adapters.parameters())
    total     = sum(p.numel() for p in model.parameters()) + trainable
    print(f"  Adapter layers   : {len(adapters)}  (hidden={hidden_size}, bottleneck={BOTTLENECK_SIZE})")
    print(f"  Trainable params : {trainable/1e6:.4f} M  ({100*trainable/total:.4f}% of total)")

    return adapters, handles, hidden_size


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
        inputs = make_batch_inputs(batch, processor)
        loss   = model(**inputs).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def save_checkpoint(adapters, hidden_size, epoch_losses):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(adapters.state_dict(), CHECKPOINT_DIR / "adapter_weights.pt")
    meta = {
        "method": "adapters",
        "config": {
            "train_size": TRAIN_SIZE, "num_epochs": NUM_EPOCHS,
            "lr": LR, "batch_size": BATCH_SIZE,
            "hidden_size": hidden_size, "bottleneck_size": BOTTLENECK_SIZE,
            "num_adapters": len(adapters),
        },
        "epoch_losses": epoch_losses,
    }
    with open(CHECKPOINT_DIR / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved → {CHECKPOINT_DIR}/")


def main():
    print("=" * 55)
    print("  Adapter Fine-tuning — BLIP-2 on VQA v2")
    print("=" * 55)
    print(f"  train={TRAIN_SIZE}  epochs={NUM_EPOCHS}  lr={LR}  batch={BATCH_SIZE}")
    print(f"  bottleneck={BOTTLENECK_SIZE}")

    print("\n[1] Loading model...")
    processor              = load_processor()
    model                  = load_model()
    model                  = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    adapters, hooks, hsize = inject_adapters(model)

    print("\n[2] Loading train dataset...")
    train_data   = VQAv2Dataset(get_fixed_train_subset()[:TRAIN_SIZE], split="train")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda x: x)
    print(f"  {len(train_data)} samples")

    print("\n[3] Training...")
    optimizer    = AdamW(adapters.parameters(), lr=LR)
    epoch_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0      = time.perf_counter()
        loss    = train_one_epoch(model, processor, train_loader, optimizer)
        elapsed = time.perf_counter() - t0
        epoch_losses.append(round(loss, 4))
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  loss={loss:.4f}  time={elapsed:.1f}s")

    print("\n[4] Saving...")
    save_checkpoint(adapters, hsize, epoch_losses)

    for h in hooks:
        h.remove()

    print("Done.")


if __name__ == "__main__":
    main()
