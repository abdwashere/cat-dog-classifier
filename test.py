import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Model Architecture (must match train.py exactly) ---
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# --- 2. Prediction ---
def predict_test_folder(folder_path, model_weight_path):
    model = CatDogCNN().to(device)
    try:
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        logging.info(f"Loaded weights from: {model_weight_path}\n")
    except FileNotFoundError:
        logging.error("Model file not found. Train first.")
        return

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(folder_path):
        logging.error(f"Folder not found: {folder_path}")
        return

    print(f"{'Filename':<30} | {'Prediction':<10} | {'Confidence'}")
    print("-" * 60)

    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.bmp')):
            try:
                img = Image.open(os.path.join(folder_path, filename)).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    label = "Dog" if predicted.item() == 1 else "Cat"
                print(f"{filename[:30]:<30} | {label:<10} | {confidence.item():.2%}")
                results.append((filename, label, confidence.item()))
            except Exception as e:
                logging.warning(f"Skipping {filename}: {e}")

    logging.info(f"\nProcessed {len(results)} images.")

# --- 3. Run ---
if __name__ == "__main__":
    TEST_DATA_DIR = r"C:\Users\Dell\Desktop\machine learning CNN Classifier\test"
    MODEL_PATH = 'newmodel.pth'
    predict_test_folder(TEST_DATA_DIR, MODEL_PATH)                                                                      

    