import os
from huggingface_hub import HfApi, snapshot_download

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/cat-dog-data"

# Download all images from dataset
snapshot_download(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    local_dir="data/new",
    token=HF_TOKEN,
)

# Sort into cats/ and dogs/ folders
import json, shutil
from pathlib import Path

cats_dir = Path("data/new/cats")
dogs_dir = Path("data/new/dogs")
cats_dir.mkdir(parents=True, exist_ok=True)
dogs_dir.mkdir(parents=True, exist_ok=True)

logs_dir = Path("data/new/logs")
if logs_dir.exists():
    for log_file in logs_dir.glob("*.json"):
        with open(log_file) as f:
            meta = json.load(f)
        src = Path("data/new/images") / meta["filename"]
        if src.exists():
            if meta["prediction"] == "cat":
                shutil.copy(src, cats_dir / meta["filename"])
            else:
                shutil.copy(src, dogs_dir / meta["filename"])

print(f"✅ Downloaded {len(list(cats_dir.glob('*')))} cats")
print(f"✅ Downloaded {len(list(dogs_dir.glob('*')))} dogs")