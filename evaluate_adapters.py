"""
Evaluate Bottleneck Adapter fine-tuned BLIP-2 checkpoint.

Usage:
  python evaluate_adapters.py

Skips automatically if results/adapters_results.json already exists.
Delete that file to force re-evaluation.
Requires: run adapters.py first to produce checkpoints/adapters/
"""

import torch
from transformers import Blip2Processor
from peft import prepare_model_for_kbit_training

from eval_utils import (
    MODEL_NAME, CHECKPOINT_DIRS, load_base_model, get_val_dataset,
    inject_and_load_adapters, evaluate, save_results,
    results_exist, checkpoint_ready, print_comparison_table,
    sample_evaluate,
)

METHOD = "adapters"


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
        print("       Run adapters.py first.")
        return

    print("=" * 55)
    print("  Evaluating: Bottleneck Adapters")
    print("=" * 55)

    print("\n[1] Loading model...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME)
    base      = load_base_model()
    # prepare_model_for_kbit_training needed to match the training setup
    base      = prepare_model_for_kbit_training(base)
    base.config.use_cache = False
    _, hooks  = inject_and_load_adapters(base, CHECKPOINT_DIRS[METHOD])
    model     = base
    model.eval()

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
    for h in hooks:
        h.remove()
    print_comparison_table()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found — running on CPU (will be very slow).")
    main()
