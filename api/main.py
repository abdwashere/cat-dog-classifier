import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from monitoring.monitor import log_prediction

# --- Load ResNet18 ---
def get_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

device = torch.device("cpu")
model  = get_model().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

# --- Transform (224x224 for ResNet) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
        img      = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor   = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]
            pred    = torch.argmax(probs).item()

        label      = "dog" if pred == 1 else "cat"
        confidence = round(probs[pred].item() * 100, 2)

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