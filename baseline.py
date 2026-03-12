"""
Step 4: Frozen baseline — zero-shot evaluation of BLIP-2 on VQA v2.

No fine-tuning is applied. This establishes the baseline metrics that all
PEFT methods (LoRA, Adapters, IA3) will be compared against:
  - VQA v2 accuracy (standard metric)
  - GPU VRAM usage
  - Inference time per sample
  - Trainable parameter count
"""

import time
import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── GPU guard ─────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("WARNING: CUDA not found — running on CPU (will be very slow).")
    print("         Use: venv\\Scripts\\python.exe baseline.py")
else:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
# ──────────────────────────────────────────────────────────────────────────────

from load_model   import load_model, load_processor, count_parameters, get_gpu_memory_gb, DEVICE
from load_dataset import (
    VQAv2Dataset, get_fixed_val_subset,
)


# ── Config ────────────────────────────────────────────────────────────────────
VAL_SIZE   = 50    # number of val samples to evaluate
BATCH_SIZE = 4     # samples per inference batch
MAX_NEW_TOKENS = 50
SEED = 42
# ──────────────────────────────────────────────────────────────────────────────


def normalize_answer(ans: str) -> str:
    """
    VQA answer normalisation for generative models.
    BLIP-2 generates full sentences (e.g. "The shirt appears to be gray.")
    but VQA scoring expects short answers (e.g. "gray").
    Steps:
      1. Take only the first sentence / clause (split on . , ; newline)
      2. Lowercase and strip punctuation
    """
    # Step 1: take first clause (split on punctuation)
    ans = re.split(r"[.,;\n]", ans)[0]
    ans = ans.lower().strip()
    ans = re.sub(r"[^\w\s]", "", ans)   # remove remaining punctuation
    ans = re.sub(r"\s+", " ", ans).strip()
    # Step 2: handle word repetition — "yes yes yes yes" → "yes"
    words = ans.split()
    if words and all(w == words[0] for w in words):
        ans = words[0]
    # Step 3: if answer starts with a number, take just the number
    # e.g. "0 people are in the photo" → "0"
    if words and words[0].isdigit():
        ans = words[0]
    return ans


def vqa_score(prediction: str, ground_truth_answers: list[str]) -> float:
    """
    Standard VQA accuracy:
      score = min(# annotators who gave this answer / 3, 1.0)
    """
    pred = normalize_answer(prediction)
    count = sum(1 for a in ground_truth_answers if normalize_answer(a) == pred)
    return min(count / 3, 1.0)


def run_inference_batch(model, processor, batch: list[dict]) -> list[str]:
    """Process a batch of samples and return decoded predictions."""
    images  = [s["image"] for s in batch]
    prompts = [f"Question: {s['question']} Answer:" for s in batch]

    # Left-padding is required for decoder-only models (OPT) during batched
    # generation — right-padding causes the model to attend to PAD tokens at
    # the end of the prompt and generate EOS or garbage immediately.
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
            min_new_tokens=1,        # force at least 1 real token
            num_beams=5,             # beam search for better quality
            length_penalty=1.0,
        )

    # Decode only the newly generated tokens (not the input prompt)
    new_tokens  = generated_ids[:, input_len:]
    predictions = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return [p.strip() for p in predictions]


def evaluate(model, processor, val_dataset: VQAv2Dataset) -> dict:
    """Run full evaluation loop and return metrics."""
    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: x,   # keep as list — images vary in size
    )

    scores      = []
    times       = []
    predictions = []

    print(f"\n  Evaluating {len(val_dataset)} samples "
          f"(batch_size={BATCH_SIZE})...")

    for batch in tqdm(loader, desc="  Inference"):
        t0 = time.perf_counter()
        preds = run_inference_batch(model, processor, batch)
        t1 = time.perf_counter()

        batch_time = (t1 - t0) / len(batch)   # per-sample time
        times.extend([batch_time] * len(batch))

        for sample, pred in zip(batch, preds):
            score = vqa_score(pred, sample["answers"])
            scores.append(score)
            predictions.append({
                "question_id" : sample["question_id"],
                "question"    : sample["question"],
                "prediction"  : pred,
                "answer"      : sample["answer"],
                "all_answers" : sample["answers"],
                "score"       : score,
            })

    accuracy = sum(scores) / len(scores) * 100
    avg_time = sum(times)  / len(times)

    return {
        "accuracy"       : accuracy,
        "avg_time_s"     : avg_time,
        "predictions"    : predictions,
        "num_samples"    : len(scores),
    }


def log_results(metrics: dict, model):
    param_counts = count_parameters(model)
    total_m      = param_counts["total"]     / 1e6
    trainable_m  = param_counts["trainable"] / 1e6

    print("\n" + "=" * 55)
    print("  Frozen Baseline Results")
    print("=" * 55)
    print(f"  VQA Accuracy         : {metrics['accuracy']:.2f}%")
    print(f"  Avg inference time   : {metrics['avg_time_s']*1000:.1f} ms/sample")
    print(f"  Samples evaluated    : {metrics['num_samples']}")
    print(f"  Total parameters     : {total_m:.2f} M")
    print(f"  Trainable parameters : {trainable_m:.2f} M")
    if DEVICE == "cuda":
        print(f"  GPU VRAM used        : {get_gpu_memory_gb():.2f} GB")

    # ── Sample predictions ───────────────────────────────────
    print("\n── Sample Predictions (first 5) ───────────────────────")
    for i, p in enumerate(metrics["predictions"][:5]):
        print(f"  [{i+1}] Q        : {p['question']}")
        print(f"       GT (MC)  : {p['answer']}")
        print(f"       GT (all) : {p['all_answers']}")
        print(f"       Pred     : {p['prediction']}")
        print(f"       Score    : {p['score']:.2f}")
        print()


def main():

    print("=" * 55)
    print("  Step 4: Frozen Baseline Evaluation")
    print("=" * 55)

    # ── Load model ───────────────────────────────────────────
    print("\n[1] Loading processor and model...")
    processor = load_processor()
    model     = load_model()
    model.eval()

    # ── Load fixed val dataset ────────────────────────────────
    print("\n[2] Loading fixed val dataset...")
    val_subset  = get_fixed_val_subset()
    val_dataset = VQAv2Dataset(val_subset, split="val")
    print(f"  Val subset: {len(val_dataset)} samples (fixed, shared across all experiments)")

    # ── Evaluate ─────────────────────────────────────────────
    print("\n[3] Running inference...")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    metrics = evaluate(model, processor, val_dataset)

    # ── Log results ──────────────────────────────────────────
    log_results(metrics, model)

    return metrics


if __name__ == "__main__":
    main()
