# CMPT 728 — PEFT Methods for BLIP-2 VQA

A systematic comparison of three Parameter-Efficient Fine-Tuning (PEFT) methods — **LoRA**, **Bottleneck Adapters**, and **IA³** — applied to [BLIP-2 (OPT-2.7B)](https://huggingface.co/Salesforce/blip2-opt-2.7b) for Visual Question Answering on the VQA v2 benchmark. All methods are trained under identical conditions and evaluated against a frozen baseline.

---

## Data

Download from: https://visualqa.org/download.html (2017 Training + Validation Data) and place under `Data/`.

| Path | Description |
|---|---|
| `Data/questions/v2_OpenEnded_mscoco_train2014_questions.json` | VQA v2 training questions |
| `Data/questions/v2_OpenEnded_mscoco_val2014_questions.json` | VQA v2 validation questions |
| `Data/annotations/v2_mscoco_train2014_annotations.json` | VQA v2 training annotations |
| `Data/annotations/v2_mscoco_val2014_annotations.json` | VQA v2 validation annotations |
| `Data/image/train2014.zip` | COCO train2014 images — **extract before use** |
| `Data/image/val2014.zip` | COCO val2014 images — **extract before use** |
| `Data/fixed_train_subset.json` | Fixed 30 000-sample training subset (shared across all methods) |
| `Data/fixed_val_subset.json` | Fixed 3 000-sample validation subset (shared across all methods) |

The fixed subsets use a common random seed so every method is trained and evaluated on exactly the same questions and images.

---

## Model

**Base model:** `Salesforce/blip2-opt-2.7b` (downloaded automatically by HuggingFace on first run)

BLIP-2 has three components:

| Component | Role | Frozen? |
|---|---|---|
| ViT-L/14 | Vision encoder | Yes |
| Q-Former | Cross-modal bridge | Yes |
| OPT-2.7B | Language model decoder | Yes (PEFT applied here) |

The model is loaded in **8-bit quantization** via `bitsandbytes`, reducing VRAM from ~7 GB (float16) to ~4 GB — making all experiments possible on a single consumer GPU. Only the frozen backbone weights are quantized; PEFT-specific trainable parameters remain in float16. Because all four methods share the same quantized backbone, any quantization-induced accuracy loss is a uniform offset that does not affect relative rankings.

---

## PEFT Methods

### 1. LoRA — `train_lora.py`
Low-Rank Adaptation injects trainable rank decomposition matrices into the attention projection layers. Only the injected matrices are updated; the backbone remains frozen.

- **Target modules:** `q_proj`, `v_proj`
- **Rank:** r = 8, alpha = 16, dropout = 0.05
- **Trainable parameters:** ~2.62 M (0.07% of backbone)
- **Checkpoint:** `checkpoints/lora/`

### 2. Bottleneck Adapters — `train_adapters.py`
Adapter modules (down-project → GELU → up-project with residual connection) are inserted after every OPT decoder layer via forward hooks. All backbone weights stay frozen.

- **Architecture:** hidden 2560 → bottleneck 128 → hidden 2560
- **Number of adapters:** 32 (one per OPT layer)
- **Trainable parameters:** ~13.1 M (0.35% of backbone)
- **Checkpoint:** `checkpoints/adapters/`

### 3. IA³ — `train_ia3.py`
Infused Adapter by Inhibiting and Amplifying Inner Activations learns element-wise scaling vectors that rescale attention keys/values and feed-forward activations. No new layers are inserted; the vectors fuse into the model at inference time.

- **Target modules:** `k_proj`, `v_proj`, `fc2`
- **Trainable parameters:** ~0.73 M (0.02% of backbone)
- **Checkpoint:** `checkpoints/ia3/`

---

## Code Structure

```
Cmpt728/
│
├── Data/                          # Dataset files (questions, annotations, images, subsets)
│
├── checkpoints/
│   ├── lora/                      # Saved LoRA adapter weights + config
│   ├── adapters/                  # Saved Bottleneck Adapter weights + train metadata
│   └── ia3/                       # Saved IA³ adapter weights + config
│
├── results/
│   ├── baseline_results.json          # Frozen baseline (normalized, 40.29%)
│   ├── baseline_no_norm_results.json  # Frozen baseline raw outputs (30.40%)
│   ├── lora_results.json              # LoRA results (47.37%)
│   ├── adapters_results.json          # Adapters results (46.13%)
│   └── ia3_results.json               # IA³ results (36.78%)
│
├── src/                           # Reusable modules (imported by train/evaluate scripts)
│   ├── dataset.py                 # VQA v2 dataset loader, fixed subset helpers
│   ├── model.py                   # BLIP-2 model + processor loader (8-bit)
│   └── eval_utils.py              # Inference, answer normalisation, VQA scoring, I/O
│
├── train/ 
│   ├── train_lora.py                  # Train LoRA — saves to checkpoints/lora/
│   ├── train_adapters.py              # Train Bottleneck Adapters — saves to checkpoints/adapters/
│   ├── train_ia3.py                   # Train IA³ — saves to checkpoints/ia3/
│
├── eval/ 
│   ├── evaluate.py                    # Evaluate any/all methods — saves to results/
│   ├── compare.py                     # Print comparison table from saved results
│
├── requirement.txt                # Python dependencies
├── Cmpt728_Final_Project_Report.pdf                # Final Report
```

### How the modules relate

```
src/dataset.py  ←──── train_*.py
src/model.py    ←──┘
                     evaluate.py ──→ results/
src/eval_utils.py ←─┘
                     compare.py
```

---

## Quick Start

```bash
# 0. Install dependencies
pip install -r requirement.txt

# 1. Extract images (do this once)
#    unzip Data/image/train2014.zip -d Data/image/
#    unzip Data/image/val2014.zip   -d Data/image/

# 2. Train a method (pick one)
python train/train_lora.py
python train/train_adapters.py
python train/train_ia3.py

# 3. Evaluate (runs all methods with checkpoints; skips if results already exist)
python eval/evaluate.py            # all methods
python eval/evaluate.py lora       # one method only

# 4. Compare results
python eval/compare.py
```

All three training scripts support **automatic resume** — if a run is interrupted, re-running the script picks up from the last completed epoch checkpoint.
