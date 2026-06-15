import os
import json
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN     = os.environ["HF_TOKEN"]
THRESHOLD    = 70.0
MIN_IMAGES   = 10

# Count new images collected
cats = list(Path("data/new/cats").glob("*"))
dogs = list(Path("data/new/dogs").glob("*"))
total = len(cats) + len(dogs)

# Read confidence logs
logs_dir = Path("data/new/logs")
confidences = []
if logs_dir.exists():
    for log_file in logs_dir.glob("*.json"):
        with open(log_file) as f:
            meta = json.load(f)
        confidences.append(meta["confidence"])

avg_conf = sum(confidences) / len(confidences) if confidences else 100

print(f"📊 New images collected: {total}")
print(f"📊 Avg confidence: {avg_conf:.2f}%")

should_retrain = avg_conf < THRESHOLD and total >= MIN_IMAGES

with open("/tmp/should_retrain.txt", "w") as f:
    f.write("true" if should_retrain else "false")

print(f"🔄 Should retrain: {should_retrain}")