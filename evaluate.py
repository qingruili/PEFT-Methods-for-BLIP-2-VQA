"""
Evaluate saved PEFT checkpoints on the fixed validation set.

Loads each method's checkpoint, runs inference, scores against ground truth,
and saves per-sample predictions + accuracy to results/<method>_results.json.

Usage:
  python evaluate.py                  # evaluate all available checkpoints
  python evaluate.py lora             # evaluate one method
  python evaluate.py lora ia3         # evaluate specific methods

Available methods: baseline, lora, adapters, ia3
"""

import sys
import torch
from transformers import Blip2Processor
from peft import PeftModel, prepare_model_for_kbit_training

from src.eval_utils import (
    MODEL_NAME, CHECKPOINT_DIRS, DEVICE, VAL_SIZE,
    load_base_model, inject_and_load_adapters,
    get_val_dataset, evaluate, save_results,
    results_exist, checkpoint_ready, print_comparison_table,
)

METHODS = ["baseline", "lora", "adapters", "ia3"]


def load_model_for_method(method: str):
    """Load the base model and apply the saved checkpoint for the given method."""
    print(f"  Loading base model...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME)
    hooks = []  # only populated for adapters

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
        base                  = load_base_model()
        base                  = prepare_model_for_kbit_training(base)
        base.config.use_cache = False
        _, hooks              = inject_and_load_adapters(base, CHECKPOINT_DIRS["adapters"])
        model                 = base
        model.eval()

    else:
        raise ValueError(f"Unknown method: {method}")

    return model, processor, hooks


def main():
    requested = sys.argv[1:] if len(sys.argv) > 1 else METHODS

    to_run = []
    for m in requested:
        if not checkpoint_ready(m):
            print(f"  [SKIP] {m} — checkpoint not ready (run train_{m}.py first)")
        elif results_exist(m):
            print(f"  [SKIP] {m} — results exist (delete results/{m}_results.json to re-run)")
        else:
            to_run.append(m)

    if not to_run:
        print("Nothing to evaluate.")
        print_comparison_table()
        return

    print(f"\nEvaluating: {to_run}")
    val_dataset = get_val_dataset()
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

        # Free GPU memory before loading the next model
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
