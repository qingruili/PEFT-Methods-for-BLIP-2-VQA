"""
Evaluate IA³ fine-tuned BLIP-2 checkpoint.

Usage:
  python evaluate_ia3.py

Skips automatically if results/ia3_results.json already exists.
Delete that file to force re-evaluation.
Requires: run ia3.py first to produce checkpoints/ia3/
"""

import torch
from transformers import Blip2Processor
from peft import PeftModel

from eval_utils import (
    MODEL_NAME, CHECKPOINT_DIRS, load_base_model, get_val_dataset,
    evaluate, save_results, results_exist, checkpoint_ready, print_comparison_table,
    sample_evaluate,
)

METHOD = "ia3"


def main():
    # ── Skip if already evaluated ─────────────────────────────────────────────
    if results_exist(METHOD):
        print(f"[SKIP] results/{METHOD}_results.json already exists.")
        print("       Delete the file to force re-evaluation.")
        print_comparison_table()
        return

    # ── Skip if checkpoint not ready ──────────────────────────────────────────
    if not checkpoint_ready(METHOD):
        print(f"[SKIP] No checkpoint found at {CHECKPOINT_DIRS[METHOD]}/")
        print("       Run ia3.py first.")
        return

    print("=" * 55)
    print("  Evaluating: IA³")
    print("=" * 55)

    print("\n[1] Loading model...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME)
    base      = load_base_model()
    model     = PeftModel.from_pretrained(base, str(CHECKPOINT_DIRS[METHOD]))
    model.eval()
    print(f"  Loaded IA³ weights from {CHECKPOINT_DIRS[METHOD]}/")

    print("\n[2] Loading val dataset...")
    val_dataset = get_val_dataset()
    print(f"  {len(val_dataset)} samples")

    #print("\n[3] Sample check (5 examples)...")
    #sample_evaluate(model, processor, val_dataset, n=5)

    # ── Full evaluation — uncomment when ready for the real run ───────────────
    print("\n[4] Running full inference...")
    metrics = evaluate(model, processor, val_dataset)
    print(f"\n  Accuracy  : {metrics['accuracy']:.2f}%")
    print(f"  ms/sample : {metrics['avg_time_ms']:.1f}")
    save_results(METHOD, metrics)
    print_comparison_table()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found — running on CPU (will be very slow).")
    main()
