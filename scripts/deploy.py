import os
from huggingface_hub import HfApi

HF_TOKEN     = os.environ["HF_TOKEN"]
HF_USERNAME  = os.environ["HF_USERNAME"]
SPACE_REPO   = f"{HF_USERNAME}/cat-dog-classifier"

api = HfApi()

api.upload_file(
    path_or_fileobj="models/best_model.pth",
    path_in_repo="best_model.pth",
    repo_id=SPACE_REPO,
    repo_type="space",
    token=HF_TOKEN,
)

print("✅ New model deployed to HuggingFace Space!")