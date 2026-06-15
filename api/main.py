import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import yaml
from monitoring.monitor import log_prediction

# --- Load config ---
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

IMG_SIZE   = config["training"]["image_size"]
MODEL_PATH = config["model"]["save_path"]

# --- Same model architecture ---
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# --- Load model ---
device = torch.device("cpu")
model  = CatDogCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- App ---
app = FastAPI(title="Cat vs Dog Classifier")

@app.get("/")
def root():
    return {"message": "Cat vs Dog Classifier API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]
            pred    = torch.argmax(probs).item()

        label      = "dog" if pred == 1 else "cat"
        confidence = round(probs[pred].item() * 100, 2)

        # ✅ log BEFORE return
        log_prediction(
            filename=file.filename,
            prediction=label,
            confidence=confidence,
            cat_prob=probs[0].item(),
            dog_prob=probs[1].item()
        )

        return JSONResponse({
            "prediction": label,
            "confidence": f"{confidence}%",
            "cat_prob":   f"{round(probs[0].item() * 100, 2)}%",
            "dog_prob":   f"{round(probs[1].item() * 100, 2)}%",
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)