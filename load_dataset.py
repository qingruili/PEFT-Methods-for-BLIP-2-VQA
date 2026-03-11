"""
Step 2: Load and verify the VQA v2 dataset from local files.

Data structure expected:
  Data/
    questions/
      v2_OpenEnded_mscoco_train2014_questions.json
      v2_OpenEnded_mscoco_val2014_questions.json
    annotations/
      v2_mscoco_train2014_annotations.json
      v2_mscoco_val2014_annotations.json
    image/
      train2014/train2014/COCO_train2014_000000XXXXXX.jpg
      val2014/val2014/COCO_val2014_000000XXXXXX.jpg
"""

import json
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent / "Data"

TRAIN_QUESTIONS  = BASE_DIR / "questions"   / "v2_OpenEnded_mscoco_train2014_questions.json"
VAL_QUESTIONS    = BASE_DIR / "questions"   / "v2_OpenEnded_mscoco_val2014_questions.json"
TRAIN_ANNOTATIONS= BASE_DIR / "annotations" / "v2_mscoco_train2014_annotations.json"
VAL_ANNOTATIONS  = BASE_DIR / "annotations" / "v2_mscoco_val2014_annotations.json"
TRAIN_IMAGE_DIR  = BASE_DIR / "image" / "train2014" / "train2014"
VAL_IMAGE_DIR    = BASE_DIR / "image" / "val2014"   / "val2014"

# Fixed subset cache — all experiments share these exact samples for fair comparison
FIXED_TRAIN_PATH = BASE_DIR / "fixed_train_subset.json"
FIXED_VAL_PATH   = BASE_DIR / "fixed_val_subset.json"

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_SIZE = 100   # subset size for trial
VAL_SIZE   = 50
SEED       = 42
# ──────────────────────────────────────────────────────────────────────────────


def load_json(path: Path) -> dict:
    print(f"  Loading {path.name} ...")
    with open(path, "r") as f:
        return json.load(f)


def build_samples(questions_path: Path, annotations_path: Path) -> list[dict]:
    """Merge questions + annotations by question_id into a flat list of samples."""
    q_data = load_json(questions_path)
    a_data = load_json(annotations_path)

    # Index annotations by question_id for O(1) lookup
    ann_index = {ann["question_id"]: ann for ann in a_data["annotations"]}

    samples = []
    for q in q_data["questions"]:
        qid = q["question_id"]
        ann = ann_index.get(qid)
        if ann is None:
            continue
        samples.append({
            "question_id"            : qid,
            "image_id"               : q["image_id"],
            "question"               : q["question"],
            "multiple_choice_answer" : ann["multiple_choice_answer"],
            "answers"                : [a["answer"] for a in ann["answers"]],
            "answer_type"            : ann["answer_type"],
            "question_type"          : ann["question_type"],
        })
    return samples


def get_image_path(image_id: int, split: str) -> Path:
    """Return the local image path given an image_id and split (train/val)."""
    if split == "train":
        return TRAIN_IMAGE_DIR / f"COCO_train2014_{image_id:012d}.jpg"
    else:
        return VAL_IMAGE_DIR   / f"COCO_val2014_{image_id:012d}.jpg"


# ── Fixed subset helpers ───────────────────────────────────────────────────────
def get_fixed_val_subset() -> list[dict]:
    """
    Return the fixed 50-sample val subset used by ALL experiments.
    Creates and saves fixed_val_subset.json on first call,
    then loads from it on every subsequent call — guaranteeing all
    experiments (baseline, LoRA, Adapters, IA3) test on the same images.
    """
    if FIXED_VAL_PATH.exists():
        print(f"  [Fixed val] Loading cached subset from {FIXED_VAL_PATH.name}")
        with open(FIXED_VAL_PATH) as f:
            return json.load(f)

    print("  [Fixed val] Creating fixed val subset for the first time...")
    import random
    random.seed(SEED)
    samples = build_samples(VAL_QUESTIONS, VAL_ANNOTATIONS)
    samples = [s for s in samples if get_image_path(s["image_id"], "val").exists()]
    random.shuffle(samples)
    subset  = samples[:VAL_SIZE]
    with open(FIXED_VAL_PATH, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"  [Fixed val] Saved {len(subset)} samples → {FIXED_VAL_PATH.name}")
    return subset


def get_fixed_train_subset() -> list[dict]:
    """
    Return the fixed 100-sample train subset used by ALL PEFT experiments.
    Creates and saves fixed_train_subset.json on first call.
    """
    if FIXED_TRAIN_PATH.exists():
        print(f"  [Fixed train] Loading cached subset from {FIXED_TRAIN_PATH.name}")
        with open(FIXED_TRAIN_PATH) as f:
            return json.load(f)

    print("  [Fixed train] Creating fixed train subset for the first time...")
    import random
    random.seed(SEED)
    samples = build_samples(TRAIN_QUESTIONS, TRAIN_ANNOTATIONS)
    samples = [s for s in samples if get_image_path(s["image_id"], "train").exists()]
    random.shuffle(samples)
    subset  = samples[:TRAIN_SIZE]
    with open(FIXED_TRAIN_PATH, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"  [Fixed train] Saved {len(subset)} samples → {FIXED_TRAIN_PATH.name}")
    return subset


# ── PyTorch Dataset ────────────────────────────────────────────────────────────
class VQAv2Dataset(Dataset):
    def __init__(self, samples: list[dict], split: str):
        self.samples = samples
        self.split   = split  # "train" or "val"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = get_image_path(sample["image_id"], self.split)
        image = Image.open(img_path).convert("RGB")
        return {
            "image"                  : image,
            "question"               : sample["question"],
            "answer"                 : sample["multiple_choice_answer"],
            "answers"                : sample["answers"],
            "question_id"            : sample["question_id"],
            "image_id"               : sample["image_id"],
            "answer_type"            : sample["answer_type"],
        }


def inspect_sample(dataset: VQAv2Dataset, idx: int = 0):
    sample = dataset[idx]
    print(f"\n── Sample #{idx} ──────────────────────────────────────")
    print(f"  question_id : {sample['question_id']}")
    print(f"  image_id    : {sample['image_id']}")
    print(f"  question    : {sample['question']}")
    print(f"  answer (MC) : {sample['answer']}")
    print(f"  all answers : {sample['answers']}")
    print(f"  answer_type : {sample['answer_type']}")
    img = sample["image"]
    assert isinstance(img, Image.Image), "Image is not a PIL Image!"
    print(f"  image size  : {img.size}  mode: {img.mode}")
    print("  [OK] Image loaded successfully as PIL.Image")


def main():
    import random
    random.seed(SEED)

    print("=" * 55)
    print("  VQA v2 — Local Dataset Loader")
    print("=" * 55)

    # ── Build full sample lists ──────────────────────────────
    print("\n[1] Building train samples...")
    train_samples = build_samples(TRAIN_QUESTIONS, TRAIN_ANNOTATIONS)
    print(f"    Total train samples: {len(train_samples):,}")

    print("\n[2] Building val samples...")
    val_samples = build_samples(VAL_QUESTIONS, VAL_ANNOTATIONS)
    print(f"    Total val samples  : {len(val_samples):,}")

    # ── Shuffle and take subsets ─────────────────────────────
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    train_subset = train_samples[:TRAIN_SIZE]
    val_subset   = val_samples[:VAL_SIZE]

    print(f"\n[3] Trial subsets → train: {len(train_subset)} | val: {len(val_subset)}")

    # ── Wrap in Dataset ──────────────────────────────────────
    train_dataset = VQAv2Dataset(train_subset, split="train")
    val_dataset   = VQAv2Dataset(val_subset,   split="val")

    # ── Inspect samples ──────────────────────────────────────
    print("\n[4] Inspecting samples...")
    inspect_sample(train_dataset, idx=0)
    inspect_sample(val_dataset,   idx=0)

    # ── Quick DataLoader test ────────────────────────────────
    print("\n[5] Testing DataLoader (batch_size=4)...")
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                        collate_fn=lambda x: x)   # keep as list (images vary in size)
    batch = next(iter(loader))
    print(f"    Batch size : {len(batch)}")
    print(f"    Fields     : {list(batch[0].keys())}")
    print("    [OK] DataLoader works correctly")

    print("\n[DONE] Dataset loaded and verified successfully.")
    return train_dataset, val_dataset


if __name__ == "__main__":
    main()
