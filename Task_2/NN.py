
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# These TIFFs are only being used as arrays here, so georeferencing warnings are
# just noise for this script.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Fixed seeds make the split and training repeatable.
np.random.seed(42)
torch.manual_seed(42)



# Build paths relative to this file so the script works from the project folder.
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"
LABELS_DIR = BASE_DIR / "labels"



# control input size and training behavior.

IMAGE_SIZE = 32
HIDDEN_SIZE = 64
NUM_CLASSES = 3
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


# The NDVI tile is reduced to one class using the average value of the tile.

LOW_VEG_THRESHOLD = 150
MEDIUM_VEG_THRESHOLD = 180




# input -> linear -> ReLU -> linear -> class scores

class SingleHiddenLayerPerceptron(nn.Module):
    """A simple neural network with one hidden layer."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # First layer: turn the long input vector into hidden features.
        self.input_layer = nn.Linear(input_size, hidden_size)
        # ReLU adds nonlinearity so the network can learn more than a straight line.
        self.activation = nn.ReLU()
        # Final layer: map hidden features to one score per class.
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x



# Convert one full NDVI tile into one class label for the whole image tile.

def label_to_class(label):
    # Take the mean NDVI so one full label raster becomes one number.
    mean_ndvi = float(np.nanmean(label))
    if mean_ndvi < LOW_VEG_THRESHOLD:
        return 0
    if mean_ndvi < MEDIUM_VEG_THRESHOLD:
        return 1
    return 2



# Build X and y by pairing every sample tile with its matching NDVI tile.
def load_dataset():
    X = []
    y = []

    for sample_path in sorted(SAMPLES_DIR.glob("*.tif*")):
        # The label uses the same filename pattern, except img_ becomes ndvi_.
        label_path = LABELS_DIR / sample_path.name.replace("img_", "ndvi_")
        if not label_path.exists():
            continue

        with rasterio.open(sample_path) as src:
            # Resize so every tile has the same shape before training.
            # Dividing by 255 keeps the values on a smaller scale.
            image = src.read(
                out_shape=(src.count, IMAGE_SIZE, IMAGE_SIZE),
                resampling=Resampling.bilinear,
            ).astype(np.float32) / 255.0

        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.float32)

        # Flatten the image because this MLP expects one long feature vector.
        X.append(image.reshape(-1))
        # Turn the full NDVI tile into one coarse class label.
        y.append(label_to_class(label))

    if not X:
        raise RuntimeError("No sample-label pairs were loaded.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)



# Split class by class so each split keeps some examples of each class.

def stratified_split(y, train_ratio=0.8):
    train_idx = []
    test_idx = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)

        # If a class has only one sample, keep it in training.
        if len(idx) == 1:
            train_idx.extend(idx)
            continue

        split = int(len(idx) * train_ratio)
        split = min(max(split, 1), len(idx) - 1)
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return train_idx, test_idx



# Keep model construction in one small helper so the training code is cleaner.

def build_model(input_size, output_size):
    return SingleHiddenLayerPerceptron(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=output_size,
    )



# Train with mini-batches, cross-entropy loss, and Adam.

def train_network(x_train, y_train, num_classes):
    # Use GPU if available, otherwise stay on CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(input_size=x_train.shape[1], output_size=num_classes).to(device)

    # Wrap NumPy arrays as tensors so DataLoader can batch them.
    dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).long(),
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # CrossEntropyLoss is standard for multiclass classification.
    criterion = nn.CrossEntropyLoss()
    # Adam is a common optimizer that usually works well without much tuning.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(model)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_seen = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Clear old gradients from the previous batch.
            optimizer.zero_grad()
            # Forward pass: get class scores from the network.
            logits = model(batch_x)
            # Compare those scores with the true class labels.
            loss = criterion(logits, batch_y)
            # Backpropagation computes gradients for all trainable weights.
            loss.backward()
            # Update the model weights.
            optimizer.step()

            batch_size = batch_x.size(0)
            running_loss += loss.item() * batch_size
            total_seen += batch_size

        # Print loss once in a while so we can see if training is improving.
        if (epoch + 1) % 10 == 0 or epoch == 0:
            mean_loss = running_loss / max(total_seen, 1)
            print(f"epoch {epoch + 1:02d}/{EPOCHS} loss {mean_loss:.4f}")

    return model, device



# Run the trained model on new data and pick the class with the highest score.

def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(x).float().to(device)
        logits = model(inputs)
        return logits.argmax(dim=1).cpu().numpy()



# Load data, split it, normalize it, train the model, and evaluate accuracy.

def main():
    X, y = load_dataset()
    train_idx, test_idx = stratified_split(y)

    if len(test_idx) == 0:
        raise RuntimeError("Stratified split produced no test samples.")

    x_train = X[train_idx]
    y_train = y[train_idx]
    x_test = X[test_idx]
    y_test = y[test_idx]

    # Learn normalization only from the training data.
    # Then reuse the same numbers for the test set.
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    print(f"samples: {len(X)}")
    print(f"class counts: {np.bincount(y, minlength=NUM_CLASSES).tolist()}")
    print(f"train: {len(x_train)}  test: {len(x_test)}")

    model, device = train_network(x_train, y_train, num_classes=NUM_CLASSES)
    preds = predict(model, x_test, device)
    accuracy = (preds == y_test).mean()

    print(f"test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
