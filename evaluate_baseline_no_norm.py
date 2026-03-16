"""
Evaluate frozen BLIP-2 baseline WITHOUT answer normalisation.

Run this to see the raw accuracy before any post-processing.
Compare with results/baseline_results.json (which uses normalisation)
to quantify how much normalisation helps.

Usage:
  python evaluate_baseline_no_norm.py
"""

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

from load_dataset import VQAv2Dataset, get_fixed_val_subset

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME     = "Salesforce/blip2-opt-2.7b"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 8
MAX_NEW_TOKENS = 20
VAL_SIZE       = 3000   # match other methods
RESULTS_DIR    = Path("results")
# ──────────────────────────────────────────────────────────────────────────────


def load_base_model():
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    model.eval()
    return model


def vqa_score_raw(prediction: str, all_answers: list[str]) -> float:
    """VQA soft accuracy with NO normalisation — raw string comparison only."""
    pred = prediction.lower().strip()   # only lowercase + strip, nothing else
    count = sum(1 for a in all_answers if a.lower().strip() == pred)
    return min(count / 3, 1.0)


def evaluate_no_norm(model, processor, val_dataset):
    loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda x: x
    )

    results     = []
    total_score = 0.0
    total_time  = 0.0

    for batch in tqdm(loader, desc="  No-norm eval"):
        images    = [s["image"]    for s in batch]
        questions = [s["question"] for s in batch]
        prompts   = [f"Question: {q} Answer:" for q in questions]

        inputs = processor(
            images=images, text=prompts,
            return_tensors="pt", padding=True
        ).to(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=1,
                do_sample=False,
                no_repeat_ngram_size=2,
                repetition_penalty=1.5,
            )
        total_time += time.perf_counter() - t0

        input_len  = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        preds      = processor.batch_decode(new_tokens, skip_special_tokens=True)
        preds      = [p.strip() for p in preds]

        for i, s in enumerate(batch):
            raw_pred = preds[i]
            score    = vqa_score_raw(raw_pred, s["answers"])
            total_score += score
            results.append({
                "question_id" : s["question_id"],
                "question"    : s["question"],
                "raw_pred"    : raw_pred,          # no normalisation applied
                "gt_answer"   : s["answer"],
                "all_answers" : s["answers"],
                "score"       : round(score, 4),
            })

    n            = len(results)
    accuracy     = (total_score / n) * 100
    avg_time_ms  = (total_time / n) * 1000

    return {
        "method"       : "baseline_no_norm",
        "accuracy"     : round(accuracy, 2),
        "avg_time_ms"  : round(avg_time_ms, 2),
        "num_samples"  : n,
        "predictions"  : results,
    }


def main():
    print("=" * 55)
    print("  Baseline (NO normalisation) — BLIP-2 frozen")
    print("=" * 55)
    print("  Purpose: compare raw accuracy vs normalised accuracy")
    print(f"  Samples : {VAL_SIZE}")

    print("\n[1] Loading model...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME)
    model     = load_base_model()

    print("\n[2] Loading val dataset...")
    val_data    = get_fixed_val_subset()[:VAL_SIZE]
    val_dataset = VQAv2Dataset(val_data, split="val")
    print(f"  {len(val_dataset)} samples")

    print("\n[3] Running inference (no normalisation)...")
    metrics = evaluate_no_norm(model, processor, val_dataset)

    print(f"\n  Raw accuracy (no norm) : {metrics['accuracy']:.2f}%")
    print(f"  ms/sample              : {metrics['avg_time_ms']:.1f}")

    # ── Load normalised baseline for comparison ────────────────────────────────
    norm_file = RESULTS_DIR / "baseline_results.json"
    if norm_file.exists():
        with open(norm_file) as f:
            norm = json.load(f)
        print(f"\n  Normalised accuracy    : {norm['accuracy']:.2f}%")
        gain = norm['accuracy'] - metrics['accuracy']
        print(f"  Normalisation gain     : +{gain:.2f}pp")
    else:
        print("\n  (run evaluate_baseline.py first to see normalised comparison)")

    # ── Save results ───────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / "baseline_no_norm_results.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved → {out_file}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found — running on CPU (will be very slow).")
    main()
