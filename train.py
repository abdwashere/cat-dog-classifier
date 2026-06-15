import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from sklearn.model_selection import train_test_split
import mlflow
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- Load config ---
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA    = config["data"]
TRAIN   = config["training"]
MODEL   = config["model"]

# --- Model ---
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

# --- Dataset ---
class CatsDogsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except:
            return torch.zeros(3, TRAIN["image_size"], TRAIN["image_size"]), self.labels[idx]

def load_images_from_folder(folder, label):
    paths, labels = [], []
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.jfif')):
            paths.append(os.path.join(folder, f))
            labels.append(label)
    logging.info(f"Found {len(paths)} images in {folder}")
    return paths, labels

# --- Train / Validate ---
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100. * correct / total

# --- Main ---
if __name__ == "__main__":
    img_size = TRAIN["image_size"]

    cats_paths, cats_labels = load_images_from_folder(DATA["cats_folder"], 0)
    dogs_paths, dogs_labels = load_images_from_folder(DATA["dogs_folder"], 1)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        cats_paths + dogs_paths, cats_labels + dogs_labels,
        test_size=TRAIN["test_size"],
        stratify=cats_labels + dogs_labels,
        random_state=TRAIN["random_state"]
    )

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(CatsDogsDataset(train_paths, train_labels, train_transform),
                              batch_size=TRAIN["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(CatsDogsDataset(val_paths, val_labels, val_transform),
                              batch_size=TRAIN["batch_size"], num_workers=2, pin_memory=True)

    os.makedirs(os.path.dirname(MODEL["save_path"]), exist_ok=True)

    # --- MLflow ---
    mlflow.set_experiment("cat-dog-classifier")

    with mlflow.start_run():
        # Log all params from config
        mlflow.log_params({
            "epochs": TRAIN["epochs"],
            "batch_size": TRAIN["batch_size"],
            "learning_rate": TRAIN["learning_rate"],
            "weight_decay": TRAIN["weight_decay"],
            "image_size": img_size,
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "train_samples": len(train_paths),
            "val_samples": len(val_paths),
        })

        model = CatDogCNN().to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(),
                               lr=TRAIN["learning_rate"],
                               weight_decay=TRAIN["weight_decay"])
        scheduler = StepLR(optimizer, step_size=7, gamma=0.5)

        best_val_acc = 0.0
        for epoch in range(1, TRAIN["epochs"] + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc     = validate(model, val_loader, criterion)
            scheduler.step()

            # Log metrics per epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            logging.info(f"Epoch [{epoch:02d}/{TRAIN['epochs']}] "
                         f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL["save_path"])
                mlflow.log_artifact(MODEL["save_path"])
                logging.info(f"  ✔ Best model saved ({best_val_acc:.2f}%)")

        # Log final summary metrics
        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
        })
        mlflow.set_tag("model_type", "CatDogCNN")
        mlflow.set_tag("device", str(device))

        logging.info(f"Done! Best Val Accuracy: {best_val_acc:.2f}%")
        logging.info(f"MLflow run complete. Run: mlflow ui to view results.")