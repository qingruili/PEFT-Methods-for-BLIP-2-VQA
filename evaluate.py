"""
Evaluate saved checkpoints on the fixed val dataset.

Loads each method's checkpoint, runs inference, scores against ground truth,
and saves per-sample predictions + accuracy to results/<method>_results.json.

Usage:
  python evaluate.py                  # evaluate all available checkpoints
  python evaluate.py lora             # evaluate one method
  python evaluate.py lora ia3         # evaluate specific methods

Available methods: lora, adapters, ia3
"""

import sys
import json
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

from load_dataset import VQAv2Dataset, get_fixed_val_subset


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "Salesforce/blip2-opt-2.7b"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 8
MAX_NEW_TOKENS = 5
VAL_SIZE       = 3000

CHECKPOINT_DIRS = {
    "lora"     : Path("checkpoints/lora"),
    "adapters" : Path("checkpoints/adapters"),
    "ia3"      : Path("checkpoints/ia3"),
}
METHODS = ["baseline", "lora", "adapters", "ia3"]
RESULTS_DIR = Path("results")
# ──────────────────────────────────────────────────────────────────────────────


# ── BottleneckAdapter (mirrors adapters.py — needed to load adapter weights) ──
class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_size, bias=True)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


def inject_and_load_adapters(model, checkpoint_dir: Path):
    """Rebuild the adapter structure and load saved weights."""
    meta_path = checkpoint_dir / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"adapter train_meta.json not found in {checkpoint_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    cfg            = meta["config"]
    hidden_size    = cfg["hidden_size"]
    bottleneck_size= cfg["bottleneck_size"]
    decoder_layers = model.language_model.model.decoder.layers

    adapters = nn.ModuleList()
    handles  = []

    def make_hook(adapter):
        def hook(module, inp, output):
            hs      = output[0]
            adapted = adapter(hs.float()).to(hs.dtype)
            return (adapted,) + output[1:]
        return hook

    for layer in decoder_layers:
        adapter = BottleneckAdapter(hidden_size, bottleneck_size).to(DEVICE)
        adapters.append(adapter)
        handles.append(layer.register_forward_hook(make_hook(adapter)))

    weights_path = checkpoint_dir / "adapter_weights.pt"
    adapters.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    # Freeze everything (eval only)
    for param in model.parameters():
        param.requires_grad = False
    for param in adapters.parameters():
        param.requires_grad = False

    print(f"  Loaded {len(adapters)} adapter layers from {weights_path.name}")
    return adapters, handles


# ── Model loading ─────────────────────────────────────────────────────────────
def load_base_model():
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        # No torch_dtype=float16 here — vision encoder LayerNorms must stay
        # in float32 to match pixel_values dtype (prepare_model_for_kbit_training
        # is not called during eval, so float16 would cause a dtype mismatch).
    )
    model.eval()
    return model


def load_model_for_method(method: str):
    """Load base model and apply the method's saved checkpoint."""
    print(f"  Loading base model...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME)

    hooks = []  # only used for adapters

    if method == "baseline":
        model = load_base_model()
        print("  Frozen baseline (no fine-tuning)")

    elif method in ("lora", "ia3"):
        ckpt_dir = CHECKPOINT_DIRS[method]
        base     = load_base_model()
        model    = PeftModel.from_pretrained(base, str(ckpt_dir))
        model.eval()
        print(f"  Loaded PEFT weights from {ckpt_dir}/")

    elif method == "adapters":
        base             = load_base_model()
        # prepare_model_for_kbit_training needed to match training setup
        base             = prepare_model_for_kbit_training(base)
        base.config.use_cache = False
        _, hooks         = inject_and_load_adapters(base, CHECKPOINT_DIRS["adapters"])
        model            = base
        model.eval()

    else:
        raise ValueError(f"Unknown method: {method}")

    return model, processor, hooks


# ── Inference & scoring ───────────────────────────────────────────────────────
def normalize_answer(ans: str) -> str:
    # Step 1: take first clause (split on punctuation)
    ans = re.split(r"[.,;\n]", ans)[0]
    # Step 1b: split on CamelCase boundary — "yesThe man in" → "yes"
    # happens when model pastes answer + next sentence without space/punctuation
    ans = re.split(r"(?<=[a-z])(?=[A-Z])", ans)[0]
    ans = ans.lower().strip()
    ans = re.sub(r"[^\w\s]", "", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    words = ans.split()
    # Step 2: handle word repetition — "yes yes yes yes" → "yes"
    if words and all(w == words[0] for w in words):
        ans = words[0]
        words = [ans]
    # Step 3: if answer starts with a number, take just the number
    # e.g. "0 people are in the photo" → "0"
    if words and words[0].isdigit():
        ans = words[0]
    # Step 4: yes/no with trailing filler — "yes it is" → "yes"
    if words and words[0] in ("yes", "no") and len(words) > 1:
        ans = words[0]
    return ans


def vqa_score(prediction: str, ground_truth_answers: list) -> float:
    pred  = normalize_answer(prediction)
    count = sum(1 for a in ground_truth_answers if normalize_answer(a) == pred)
    return min(count / 3, 1.0)


def run_inference_batch(model, processor, batch):
    images  = [s["image"] for s in batch]
    prompts = [f"Question: {s['question']} Answer:" for s in batch]

    processor.tokenizer.padding_side = "left"
    inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=1,
            do_sample=False,          # greedy decoding — 5x faster than beam search
            no_repeat_ngram_size=2,
            repetition_penalty=1.5,
        )

    new_tokens = generated_ids[:, input_len:]
    preds      = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return [p.strip() for p in preds]


def evaluate(model, processor, val_dataset):
    loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda x: x,
    )

    scores      = []
    times       = []
    predictions = []

    for batch in tqdm(loader, desc="  Inference"):
        t0    = time.perf_counter()
        preds = run_inference_batch(model, processor, batch)
        t1    = time.perf_counter()

        per_sample = (t1 - t0) / len(batch)
        times.extend([per_sample] * len(batch))

        for sample, pred in zip(batch, preds):
            score = vqa_score(pred, sample["answers"])
            scores.append(score)
            predictions.append({
                "question_id" : sample["question_id"],
                "question"    : sample["question"],
                "prediction"  : pred,
                "gt_answer"   : sample["answer"],
                "all_answers" : sample["answers"],
                "score"       : score,
            })

    return {
        "accuracy"    : sum(scores) / len(scores) * 100,
        "avg_time_ms" : sum(times)  / len(times)  * 1000,
        "num_samples" : len(scores),
        "predictions" : predictions,
    }


# ── Save / print ──────────────────────────────────────────────────────────────
def save_results(method: str, metrics: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{method}_results.json"
    with open(path, "w") as f:
        json.dump({"method": method, **metrics}, f, indent=2)
    print(f"  Results saved → {path}")


def print_comparison_table():
    """Print a summary table from all available result files."""
    all_methods = ["baseline", "lora", "adapters", "ia3"]
    rows = []
    for m in all_methods:
        p = RESULTS_DIR / f"{m}_results.json"
        if p.exists():
            with open(p) as f:
                r = json.load(f)
            rows.append((m, r["accuracy"], r["avg_time_ms"], r["num_samples"]))

    if not rows:
        return

    print("\n" + "=" * 55)
    print("  Comparison Table")
    print("=" * 55)
    print(f"  {'Method':<12} {'Accuracy':>10}  {'ms/sample':>10}  {'Samples':>8}")
    print(f"  {'-'*12} {'-'*10}  {'-'*10}  {'-'*8}")
    for method, acc, ms, n in rows:
        print(f"  {method:<12} {acc:>9.2f}%  {ms:>9.1f}ms  {n:>8}")
    print("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Determine which methods to evaluate
    requested = sys.argv[1:] if len(sys.argv) > 1 else METHODS

    # Filter to only methods whose checkpoint files are actually present
    def checkpoint_ready(method):
        if method == "baseline":
            return True  # no checkpoint needed
        ckpt = CHECKPOINT_DIRS[method]
        if method == "adapters":
            return (ckpt / "adapter_weights.pt").exists() and (ckpt / "train_meta.json").exists()
        else:  # lora, ia3 — PEFT saves adapter_config.json
            return (ckpt / "adapter_config.json").exists()

    def results_exist(method):
        return (RESULTS_DIR / f"{method}_results.json").exists()

    to_run = []
    for m in requested:
        if not checkpoint_ready(m):
            print(f"  [SKIP] {m} — checkpoint not ready (run {m}.py first)")
        elif results_exist(m):
            print(f"  [SKIP] {m} — results already exist (delete results/{m}_results.json to re-run)")
        else:
            to_run.append(m)

    if not to_run:
        print("No checkpoints found. Run the training scripts first.")
        return

    print(f"\nEvaluating: {to_run}")

    # Load fixed val dataset once
    val_dataset = VQAv2Dataset(get_fixed_val_subset()[:VAL_SIZE], split="val")
    print(f"Val dataset: {len(val_dataset)} samples\n")

    for method in to_run:
        print("=" * 55)
        print(f"  [{method.upper()}]")
        print("=" * 55)

        model, processor, hooks = load_model_for_method(method)

        metrics = evaluate(model, processor, val_dataset)
        print(f"  Accuracy    : {metrics['accuracy']:.2f}%")
        print(f"  ms/sample   : {metrics['avg_time_ms']:.1f}")

        save_results(method, metrics)

        # Free GPU memory before loading next model
        for h in hooks:
            h.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print()

    print_comparison_table()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found — running on CPU (will be very slow).")
    main()
