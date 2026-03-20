

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
#add matplot library for graphics, change labels
#jupiter notebook


samples_dir = r"Learning_Path/samples"
labels_dir  = r"Learning_Path/labels"

MAX_IMAGES_FOR_TRAINING = 100      # Only use the first N tiles so training stays manageable.
MAX_SAMPLES_FOR_SVM = 9000000      # If the pixel table is huge, randomly cap it at this size.
SVM_KERNEL = "rbf"                 # "linear" is faster, "rbf" usually captures more complex boundaries.
AUTO_CLOSE_SECONDS = None          # Set a number if you want plots to close automatically.

# The labels are NDVI-like values, so we bin them into 4 simpler classes.
# 0 = non-veg, 1 = low, 2 = medium, 3 = dense
BIN_0 = 80
BIN_1 = 130
BIN_2 = 180


def maybe_show():
    """Show a plot now, or briefly show it and close it if auto-close is enabled."""
    if AUTO_CLOSE_SECONDS is None:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(AUTO_CLOSE_SECONDS)
        plt.close()


def make_rgb_display(image_3band):
    
    rgb = image_3band[:3].astype(np.float32)
    rgb = np.transpose(rgb, (1, 2, 0))  # (H,W,3)

    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        low, high = np.percentile(rgb[:, :, c], (2, 98))
        if high > low:
            out[:, :, c] = np.clip((rgb[:, :, c] - low) / (high - low), 0, 1)
        else:
            out[:, :, c] = 0
    return out


def get_class_cmap():
    colors = [
        (0.15, 0.15, 0.15),  # 0 non-veg (dark gray)
        (0.25, 0.60, 0.90),  # 1 low (blue)
        (0.25, 0.80, 0.35),  # 2 medium (green)
        (0.95, 0.75, 0.15),  # 3 dense (gold)
    ]
    cmap = ListedColormap(colors, name="veg_classes")
    boundaries = np.arange(-0.5, 4.5, 1)  # -0.5,0.5,1.5,2.5,3.5
    norm = BoundaryNorm(boundaries, cmap.N)
    class_names = ["Non-veg (0)", "Low (1)", "Medium (2)", "Dense (3)"]
    short_names = ["Non-veg", "Low", "Medium", "Dense"]
    return cmap, norm, class_names, short_names


def show_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = cm.max() * 0.6 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    maybe_show()


def show_metric_summary(accuracy, precision, recall, cm, short_names):
    """Plot the main test scores and class-by-class recall."""
    # The diagonal contains the correct predictions for each class.
    class_totals = cm.sum(axis=1)
    class_recall = np.divide(
        np.diag(cm),
        class_totals,
        out=np.zeros(len(short_names), dtype=np.float32),
        where=class_totals > 0,
    )

    class_colors = list(get_class_cmap()[0].colors)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left side: one bar per overall metric.
    metric_names = ["Accuracy", "Precision", "Recall"]
    metric_values = [accuracy, precision, recall]
    metric_colors = ["#4c72b0", "#55a868", "#c44e52"]
    axes[0].bar(metric_names, metric_values, color=metric_colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall Test Metrics")
    axes[0].grid(axis="y", alpha=0.3)

    for i, value in enumerate(metric_values):
        axes[0].text(i, min(value + 0.02, 1.0), f"{value:.3f}", ha="center", va="bottom")

    # Right side: recall for each vegetation class.
    axes[1].bar(short_names, class_recall, color=class_colors)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Per-Class Recall")
    axes[1].grid(axis="y", alpha=0.3)

    for i, value in enumerate(class_recall):
        axes[1].text(i, min(value + 0.02, 1.0), f"{value:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    maybe_show()



def ndvi_to_class(label):
    """Convert raw NDVI values into 4 classes for every pixel in the label tile."""
    classes = np.zeros(label.shape, dtype=np.uint8)
    classes[label < BIN_0] = 0
    classes[(label >= BIN_0) & (label < BIN_1)] = 1
    classes[(label >= BIN_1) & (label < BIN_2)] = 2
    classes[label >= BIN_2] = 3
    return classes



images = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith((".tif", ".tiff"))])
if not images:
    raise FileNotFoundError("No .tif/.tiff files found in samples folder.")

print("Available images (showing first 60):")
for i, img in enumerate(images[:60]):
    print(i, "-", img)
print("... total images:", len(images))

idx = int(input("\nSelect image index for visualization: "))
if idx < 0 or idx >= len(images):
    raise ValueError("Invalid image index.")

image_name = images[idx]
image_path = os.path.join(samples_dir, image_name)
label_name = image_name.replace("img_", "ndvi_")
label_path = os.path.join(labels_dir, label_name)

print("\nLoading image:", image_name)
print("Loading label:", label_name)

with rasterio.open(image_path) as src:
    image = src.read()

with rasterio.open(label_path) as src:
    label = src.read(1)

# The chosen band is just for display. The full set of bands is still used later.
print("\nThis image has", image.shape[0], "bands (0 to", image.shape[0] - 1, ")")
band = int(input("Choose band index to display: "))
if band < 0 or band >= image.shape[0]:
    print("Invalid band. Using band 0.")
    band = 0


cmap_cls, norm_cls, class_names, short_names = get_class_cmap()

fig = plt.figure(figsize=(16, 5))
fig.suptitle(f"Sample Tile: {image_name}", fontsize=12)

# Show an RGB-style view if at least 3 bands are available.
ax1 = plt.subplot(1, 4, 1)
if image.shape[0] >= 3:
    rgb_disp = make_rgb_display(image[:3])
    ax1.imshow(rgb_disp)
    ax1.set_title("RGB (2–98% stretch)")
else:
    ax1.imshow(image[0], cmap="gray")
    ax1.set_title("Not enough bands for RGB")
ax1.axis("off")

# Show one individual band with contrast stretching.
ax2 = plt.subplot(1, 4, 2)
band_img = image[band].astype(np.float32)
b_lo, b_hi = np.percentile(band_img, (2, 98))
ax2.imshow(band_img, cmap="gray", vmin=b_lo, vmax=b_hi)
ax2.set_title(f"Band {band + 1} (2–98% stretch)")
ax2.axis("off")

# Show the raw NDVI label values.
ax3 = plt.subplot(1, 4, 3)
ndvi_img = label.astype(np.float32)
n_lo, n_hi = np.percentile(ndvi_img, (2, 98))
im3 = ax3.imshow(ndvi_img, cmap="viridis", vmin=n_lo, vmax=n_hi)
ax3.set_title("NDVI Label")
ax3.axis("off")
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label("NDVI (scaled uint8)")

# Show the simplified class map that the model will actually learn.
ax4 = plt.subplot(1, 4, 4)
class_label = ndvi_to_class(label)
im4 = ax4.imshow(class_label, cmap=cmap_cls, norm=norm_cls)
ax4.set_title("Vegetation Classes (binned)")
ax4.axis("off")
cbar4 = plt.colorbar(im4, ax=ax4, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
cbar4.ax.set_yticklabels(class_names)

print("\nImage shape (bands,H,W):", image.shape)
print("Label shape (H,W):", label.shape)
print("Image min/max:", np.min(image), np.max(image))
print("Label (NDVI) min/max:", np.min(label), np.max(label))
print("Class labels present:", np.unique(class_label))
print(f"Bins used: <{BIN_0}=0, [{BIN_0},{BIN_1})=1, [{BIN_1},{BIN_2})=2, >= {BIN_2}=3")

plt.tight_layout()
maybe_show()



X_list = []
y_list = []

# Limiting the number of tiles keeps the script from becoming painfully slow.
train_images = images[:MAX_IMAGES_FOR_TRAINING]
print(f"\nBuilding dataset from {len(train_images)} image-label pairs...", flush=True)

for k, img_name in enumerate(train_images, start=1):
    img_path = os.path.join(samples_dir, img_name)
    lab_name = img_name.replace("img_", "ndvi_")
    lab_path = os.path.join(labels_dir, lab_name)

    if not os.path.exists(lab_path):
        print("Skipping, label not found for:", img_name, flush=True)
        continue

    with rasterio.open(img_path) as src:
        img = src.read()

    with rasterio.open(lab_path) as src:
        ndvi = src.read(1)

    # Turn the label image into integer classes the SVM can learn.
    y_cls = ndvi_to_class(ndvi)

    bands, h, w = img.shape

    # This is the key reshape:
    # every pixel becomes one row, and the band values become the columns.
    X = img.reshape(bands, -1).T.astype(np.float32)
    y = y_cls.reshape(-1).astype(np.uint8)

    # Drop rows with invalid band values so the SVM only sees real numbers.
    valid_mask = np.ones(X.shape[0], dtype=bool)
    for b in range(X.shape[1]):
        valid_mask &= np.isfinite(X[:, b])

    X = X[valid_mask]
    y = y[valid_mask]

    X_list.append(X)
    y_list.append(y)

    if k % 5 == 0 or k == len(train_images):
        print(f"  Progress {k}/{len(train_images)} | latest tile samples: {X.shape[0]}", flush=True)

if len(X_list) == 0:
    raise FileNotFoundError("No valid image-label pairs were found.")

# Stack all tiles together into one big pixel table.
X = np.vstack(X_list)
y = np.concatenate(y_list)

print("\nFinal dataset shape (before sampling):")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Unique classes:", np.unique(y), flush=True)



counts_all = np.bincount(y, minlength=4)
plt.figure(figsize=(6, 4))
plt.bar([0, 1, 2, 3], counts_all)
plt.xticks([0, 1, 2, 3], short_names)
plt.title("Class Distribution (before sampling)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
maybe_show()

if len(y) > MAX_SAMPLES_FOR_SVM:
    print(f"\nDataset is large. Randomly sampling {MAX_SAMPLES_FOR_SVM} pixels for SVM...", flush=True)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(y), size=MAX_SAMPLES_FOR_SVM, replace=False)
    X = X[indices]
    y = y[indices]

print("Dataset used for SVM:", X.shape, y.shape, flush=True)


counts_s = np.bincount(y, minlength=4)
plt.figure(figsize=(6, 4))
plt.bar([0, 1, 2, 3], counts_s)
plt.xticks([0, 1, 2, 3], short_names)
plt.title("Class Distribution (after sampling)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
maybe_show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain/Test split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape, flush=True)



print("\nTraining SVM classifier...", flush=True)
model = SVC(kernel=SVM_KERNEL, random_state=42)
model.fit(X_train, y_train)

# Predict the classes for the held-out test pixels.
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nSVM Evaluation Results:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)

print("\nClassification Report:")
labels_all = [0, 1, 2, 3]
print(classification_report(
    y_test, y_pred,
    labels=labels_all,
    target_names=short_names,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred, labels=labels_all)
print("Confusion Matrix (raw):")
print(cm)



show_confusion_matrix(cm, short_names, title="SVM Confusion Matrix")

# Plot one more graph so the main scores are easier to read quickly.
show_metric_summary(accuracy, precision, recall, cm, short_names)



print("\nPredicting a full tile for visualization...", flush=True)

chosen_image_path = os.path.join(samples_dir, images[idx])
chosen_label_name = images[idx].replace("img_", "ndvi_")
chosen_label_path = os.path.join(labels_dir, chosen_label_name)

with rasterio.open(chosen_image_path) as src:
    img0 = src.read()

with rasterio.open(chosen_label_path) as src:
    ndvi0 = src.read(1)

true_cls = ndvi_to_class(ndvi0)

bands0, h0, w0 = img0.shape

# Use the same pixel-table layout here that we used while training.
X0 = img0.reshape(bands0, -1).T.astype(np.float32)

# Predict one class per pixel, then turn the flat output back into a 2D map.
pred_cls_flat = model.predict(X0)
pred_cls = pred_cls_flat.reshape(h0, w0)

# White pixels in this map are the places where the prediction was wrong.
error_map = (pred_cls != true_cls).astype(np.uint8)

fig = plt.figure(figsize=(16, 5))
fig.suptitle("SVM Classification on Selected Tile", fontsize=12)

ax1 = plt.subplot(1, 3, 1)
im1 = ax1.imshow(true_cls, cmap=cmap_cls, norm=norm_cls)
ax1.set_title("True Vegetation Class")
ax1.axis("off")

ax2 = plt.subplot(1, 3, 2)
im2 = ax2.imshow(pred_cls, cmap=cmap_cls, norm=norm_cls)
ax2.set_title("Predicted Vegetation Class")
ax2.axis("off")

ax3 = plt.subplot(1, 3, 3)
ax3.imshow(error_map, cmap="gray")
ax3.set_title("Error Map (white = wrong)")
ax3.axis("off")

# Shared class colorbar
cbar = plt.colorbar(im2, ax=[ax1, ax2], ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(class_names)

plt.tight_layout()
maybe_show()
