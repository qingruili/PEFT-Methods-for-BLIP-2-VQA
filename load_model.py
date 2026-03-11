"""
Step 3: Load BLIP-2 (OPT-2.7B) model and processor from HuggingFace.

- Loads Blip2Processor (handles image preprocessing + text tokenization)
- Loads Blip2ForConditionalGeneration in 8-bit (GPU) or float32 (CPU)
  * 8-bit via bitsandbytes keeps VRAM ~4-5 GB (safe for 8 GB cards)
  * float16 would need ~7 GB — too tight for training headroom
- Logs total/trainable parameters and memory usage
- Runs a single sanity-check inference
"""

import torch
import psutil
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "Salesforce/blip2-opt-2.7b"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_IN_8BIT = DEVICE == "cuda"   # 8-bit quantization only works on GPU
# ──────────────────────────────────────────────────────────────────────────────


def get_ram_usage_gb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 3


def get_gpu_memory_gb() -> float:
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / 1024 ** 3
    return 0.0


def count_parameters(model) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def load_processor():
    print(f"  Loading processor from {MODEL_NAME} ...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME)
    print("  [OK] Processor loaded")
    return processor


def load_model():
    mode = "8-bit (GPU)" if LOAD_IN_8BIT else "float32 (CPU)"
    print(f"  Loading model from {MODEL_NAME} ...")
    print(f"  Device: {DEVICE.upper()}  |  mode: {mode}")
    ram_before = get_ram_usage_gb()

    if LOAD_IN_8BIT:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        ).to(DEVICE)

    model.eval()

    ram_after = get_ram_usage_gb()
    print(f"  [OK] Model loaded")
    print(f"  RAM usage          : {ram_after - ram_before:.2f} GB")
    if DEVICE == "cuda":
        print(f"  GPU VRAM used      : {get_gpu_memory_gb():.2f} GB")
    return model


def log_parameter_counts(model):
    counts = count_parameters(model)
    total_m     = counts["total"]     / 1e6
    trainable_m = counts["trainable"] / 1e6
    print(f"\n── Parameter Counts ───────────────────────────────────")
    print(f"  Total parameters     : {total_m:>10.2f} M")
    print(f"  Trainable parameters : {trainable_m:>10.2f} M")
    print(f"  Frozen parameters    : {total_m - trainable_m:>10.2f} M")


def log_model_components(model):
    print(f"\n── Model Components ───────────────────────────────────")
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters()) / 1e6
        print(f"  {name:<30} {params:>8.2f} M params")


def sanity_check_inference(model, processor):
    """Run a single forward pass to verify the model works end-to-end."""
    print(f"\n── Sanity Check Inference ─────────────────────────────")

    # Create a dummy image (solid colour) to avoid needing a real image here
    dummy_image  = Image.new("RGB", (224, 224), color=(128, 128, 128))
    dummy_question = "What color is this image?"

    inputs = processor(
        images=dummy_image,
        text=dummy_question,
        return_tensors="pt",
    ).to(DEVICE)

    print(f"  Input pixel_values shape : {inputs['pixel_values'].shape}")
    print(f"  Input ids shape          : {inputs['input_ids'].shape}")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=10,
        )

    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"  Question : {dummy_question}")
    print(f"  Answer   : {answer}")
    print("  [OK] Inference completed successfully")


def main():
    print("=" * 55)
    print("  BLIP-2 (OPT-2.7B) — Model Loader")
    print("=" * 55)

    mode = "8-bit quantized" if LOAD_IN_8BIT else "float32"
    print(f"\n[1] Device setup: {DEVICE.upper()}  |  mode: {mode}")

    print("\n[2] Loading processor...")
    processor = load_processor()

    print("\n[3] Loading model...")
    model = load_model()

    log_parameter_counts(model)
    log_model_components(model)

    print("\n[4] Running sanity check...")
    sanity_check_inference(model, processor)

    print("\n[DONE] Model and processor loaded and verified successfully.")
    return model, processor


if __name__ == "__main__":
    model, processor = main()
