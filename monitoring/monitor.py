import json
import os
from datetime import datetime

LOG_FILE = "monitoring/predictions.json"

def log_prediction(filename: str, prediction: str, confidence: float, cat_prob: float, dog_prob: float):
    os.makedirs("monitoring", exist_ok=True)
    
    record = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "prediction": prediction,
        "confidence": confidence,
        "cat_prob": cat_prob,
        "dog_prob": dog_prob,
    }

    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)

    logs.append(record)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def get_logs():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        return json.load(f)