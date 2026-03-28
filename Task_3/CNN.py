from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# Paths
TASK_DIR = Path(__file__).resolve().parent
BASE_DIR = TASK_DIR.parent
SAMPLES_DIR = BASE_DIR / "samples"
LABELS_DIR = BASE_DIR / "labels"
OUTPUT_DIR = TASK_DIR / "outputs"

# Simple settings
IMAGE_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 4
SEED = 42

# NDVI thresholds for 3 classes
LOW_THRESHOLD = 150
MEDIUM_THRESHOLD = 180
CLASS_NAMES = ["low", "medium", "high"]


np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def label_to_class(label_tile):
    mean_ndvi = float(np.nanmean(label_tile))

    if mean_ndvi < LOW_THRESHOLD:
        return 0
    if mean_ndvi < MEDIUM_THRESHOLD:
        return 1
    return 2


# -----------------------------------------------------------------------------
# Load the TIFF data
# -----------------------------------------------------------------------------

images = []
labels = []

for sample_path in sorted(SAMPLES_DIR.glob("*.tif*")):
    label_path = LABELS_DIR / sample_path.name.replace("img_", "ndvi_")

    if not label_path.exists():
        continue

    with rasterio.open(sample_path) as src:
        image = src.read(
            out_shape=(src.count, IMAGE_SIZE, IMAGE_SIZE),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    with rasterio.open(label_path) as src:
        label = src.read(1).astype(np.float32)

    image = image / 255.0
    image = np.nan_to_num(image)

    images.append(image)
    labels.append(label_to_class(label))

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int64)

if len(images) == 0:
    raise RuntimeError("No image/label pairs were loaded.")

print(f"Total images: {len(images)}")
print(f"Class counts: {dict(zip(CLASS_NAMES, np.bincount(labels, minlength=3)))}")


# -----------------------------------------------------------------------------
# Split into train / validation / test
# -----------------------------------------------------------------------------

train_images, temp_images, train_labels, temp_labels = train_test_split(
    images,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=SEED,
)

val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SEED,
)

print(f"Train: {len(train_images)}")
print(f"Validation: {len(val_images)}")
print(f"Test: {len(test_images)}")


# Normalize using only the training set
mean = train_images.mean(axis=(0, 2, 3), keepdims=True)
std = train_images.std(axis=(0, 2, 3), keepdims=True)
std[std == 0] = 1.0

train_images = (train_images - mean) / std
val_images = (val_images - mean) / std
test_images = (test_images - mean) / std


# Convert to PyTorch datasets
train_data = TensorDataset(
    torch.tensor(train_images, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long),
)
val_data = TensorDataset(
    torch.tensor(val_images, dtype=torch.float32),
    torch.tensor(val_labels, dtype=torch.long),
)
test_data = TensorDataset(
    torch.tensor(test_images, dtype=torch.float32),
    torch.tensor(test_labels, dtype=torch.long),
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# -----------------------------------------------------------------------------
# CNN model
# -----------------------------------------------------------------------------

class VegetationCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = VegetationCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
best_model_path = OUTPUT_DIR / "basic_cnn.pth"

best_val_loss = float("inf")
wait = 0

for epoch in range(EPOCHS):
    net.train()
    running_loss = 0.0

    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        outputs = net(batch_images)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    net.eval()
    val_loss_total = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = net(batch_images)
            loss = loss_fn(outputs, batch_labels)
            val_loss_total += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    val_loss = val_loss_total / len(val_loader)
    val_accuracy = correct / total

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        torch.save(net.state_dict(), best_model_path)
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping.")
            break


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

net = VegetationCNN().to(device)
net.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
net.eval()

all_true = []
all_pred = []

with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.to(device)

        outputs = net(batch_images)
        _, predicted = torch.max(outputs, 1)

        all_true.extend(batch_labels.numpy())
        all_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(all_true, all_pred)
precision = precision_score(all_true, all_pred, average="macro", zero_division=0)
recall = recall_score(all_true, all_pred, average="macro", zero_division=0)
f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

print("\nTest Results")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

with open(OUTPUT_DIR / "basic_results.txt", "w") as file:
    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall   : {recall:.4f}\n")
    file.write(f"F1-Score : {f1:.4f}\n")

print(f"\nSaved model: {best_model_path}")
