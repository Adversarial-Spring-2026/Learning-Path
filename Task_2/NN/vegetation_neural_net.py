import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             classification_report, confusion_matrix)
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SAMPLES = "/Users/m4fanta/projects/uprm-projects/research/pandahat-adversarial/Learning-Path/samples"   # directory containing RGB .tiff images
LABELS  = "/Users/m4fanta/projects/uprm-projects/research/pandahat-adversarial/Learning-Path/labels"    # directory containing NDVI .tiff images
PATCH   = 16           # pixel width/height of each sampled patch (16x16)
THR     = 0.3          # NDVI threshold above which a pixel is considered vegetation
N       = 2000         # number of random patches sampled per image
HIDDEN  = 128           # number of neurons in the hidden layer
EPOCHS  = 1000           # number of full passes through the training data
BATCH   = 256          # number of patches per gradient update
LR      = 0.01         # learning rate for SGD

np.random.seed(42)
torch.manual_seed(42)


def load_pairs():
    pairs = []
    for f in sorted(os.listdir(SAMPLES)):
        if not f.endswith(".tiff"):
            continue
        ndvi = os.path.join(LABELS, f.replace("_img_", "_ndvi_"))
        if os.path.exists(ndvi):
            pairs.append((os.path.join(SAMPLES, f), ndvi))
    return pairs


def extract_patches(img_path, ndvi_path):
    rgb = np.array(Image.open(img_path)).astype(np.float32)
    rgb = (rgb - rgb.mean()) / (rgb.std() + 1e-8)

    ndvi = np.array(Image.open(ndvi_path)).astype(np.float32)
    if ndvi.max() > 1 or ndvi.min() < -1:
        ndvi = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min()) * 2 - 1

    H, W = rgb.shape[:2]
    half = PATCH // 2
    ys = np.random.randint(half, H - half, N)
    xs = np.random.randint(half, W - half, N)

    patches = [rgb[y-half:y+half, x-half:x+half].transpose(2, 0, 1).ravel() for y, x in zip(ys, xs)]
    labels  = (ndvi[ys, xs] > THR * 2 - 1).astype(np.int64)
    return np.stack(patches), labels


def build_tensors(pairs):
    all_X, all_y = [], []
    for img, ndvi in pairs:
        X, y = extract_patches(img, ndvi)
        all_X.append(X)
        all_y.append(y)
    return torch.tensor(np.concatenate(all_X)), torch.tensor(np.concatenate(all_y))


def eval_metrics(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1).numpy()
    true = y.numpy()
    acc  = accuracy_score(true, preds)
    prec = precision_score(true, preds, average="macro", zero_division=0)
    rec  = recall_score(true, preds, average="macro", zero_division=0)
    return acc, prec, rec, preds


# --- Split: 70% train, 10% val, 20% test (by whole image to avoid leakage) ---
pairs     = load_pairs()
n         = len(pairs)
test_cut  = max(1, int(n * 0.2))
val_cut   = max(1, int(n * 0.1))

train_pairs = pairs[:-(test_cut + val_cut)]
val_pairs   = pairs[-(test_cut + val_cut):-test_cut]
test_pairs  = pairs[-test_cut:]

X_tr, y_tr = build_tensors(train_pairs)
X_val, y_val = build_tensors(val_pairs)
X_te, y_te = build_tensors(test_pairs)

# --- Weighted loss instead of oversampling: handles imbalance without duplicating data ---
n_veg     = (y_tr == 1).sum().float()
n_nonveg  = (y_tr == 0).sum().float()
weights   = torch.tensor([n_veg / (n_veg + n_nonveg), n_nonveg / (n_veg + n_nonveg)])

train_dl = DataLoader(TensorDataset(X_tr, y_tr), BATCH, shuffle=True)

model   = nn.Sequential(nn.Linear(X_tr.shape[1], HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 2))
opt     = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
loss_fn = nn.CrossEntropyLoss(weight=weights)

# --- Training loop: track BOTH train and val metrics so we see real fluctuation ---
history = {"loss": [], "val_loss": [], "acc": [], "val_acc": [], "prec": [], "val_prec": [], "rec": [], "val_rec": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0
    for X, y in train_dl:
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        total += loss.item()

    avg_loss = total / len(train_dl)

    # Val loss
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val).item()

    acc, prec, rec, _       = eval_metrics(model, X_tr, y_tr)
    val_acc, val_prec, val_rec, _ = eval_metrics(model, X_val, y_val)

    history["loss"].append(avg_loss)
    history["val_loss"].append(val_loss)
    history["acc"].append(acc);   history["val_acc"].append(val_acc)
    history["prec"].append(prec); history["val_prec"].append(val_prec)
    history["rec"].append(rec);   history["val_rec"].append(val_rec)

    print(f"epoch {epoch:>3}  loss {avg_loss:.4f} | val_loss {val_loss:.4f} | "
          f"acc {acc*100:.1f}% / {val_acc*100:.1f}%  "
          f"prec {prec*100:.1f}% / {val_prec*100:.1f}%  "
          f"rec {rec*100:.1f}% / {val_rec*100:.1f}%")

# --- Test evaluation ---
acc, prec, rec, preds = eval_metrics(model, X_te, y_te)
true = y_te.numpy()

print(f"\nTest set results:")
print(f"  Accuracy  {acc*100:.2f}%")
print(f"  Precision {prec*100:.2f}%")
print(f"  Recall    {rec*100:.2f}%")
print(classification_report(true, preds, target_names=["Non-Veg", "Veg"]))

# =============================================================================
# FIGURE 1 — Learning curves (train vs val so fluctuation is visible)
# =============================================================================
epochs = range(1, EPOCHS + 1)
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("Training History — Train (solid) vs Validation (dashed)", fontsize=13)

def plot_curve(ax, metric, title, ylabel, color):
    ax.plot(epochs, history[metric],         color=color, linewidth=1.8, label="Train")
    ax.plot(epochs, history[f"val_{metric}"], color=color, linewidth=1.8,
            linestyle="--", alpha=0.7, label="Val")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_xlim(1, EPOCHS); ax.grid(alpha=0.3); ax.legend(fontsize=8)

plot_curve(axes[0, 0], "loss",  "Cross-Entropy Loss",      "Loss",          "#c44e52")
plot_curve(axes[0, 1], "acc",   "Accuracy",                "Accuracy",      "#4c72b0")
plot_curve(axes[1, 0], "prec",  "Precision (macro avg)",   "Precision",     "#55a868")
plot_curve(axes[1, 1], "rec",   "Recall (macro avg)",      "Recall",        "#dd8452")

# Convert acc/prec/rec axes to percentages
for ax in axes.flat[1:]:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()

# =============================================================================
# FIGURE 2 — Test evaluation (bar chart + confusion matrix)
# =============================================================================
cm = confusion_matrix(true, preds)
tn, fp, fn, tp = cm.ravel()

fig = plt.figure(figsize=(13, 5))
fig.suptitle("Test Set Evaluation", fontsize=14)
gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
scores  = [acc * 100, prec * 100, rec * 100]
blabels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)"]
colors  = ["#4c72b0", "#55a868", "#c44e52"]
bars    = ax1.bar(blabels, scores, color=colors, width=0.45, edgecolor="white")
for bar, val in zip(bars, scores):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 115); ax1.set_ylabel("Score (%)"); ax1.grid(axis="y", alpha=0.3)

ax2 = fig.add_subplot(gs[1])
im = ax2.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(["Predicted\nNon-Veg", "Predicted\nVeg"], fontsize=10)
ax2.set_yticklabels(["Actual\nNon-Veg", "Actual\nVeg"], fontsize=10)
ax2.set_title("Confusion Matrix")
for row, col, text in [(0,0,f"TN\n{tn:,}"),(0,1,f"FP\n{fp:,}"),(1,0,f"FN\n{fn:,}"),(1,1,f"TP\n{tp:,}")]:
    color = "white" if cm[row, col] > cm.max() / 2 else "black"
    ax2.text(col, row, text, ha="center", va="center", fontsize=10, color=color)

plt.tight_layout()
plt.savefig("test_evaluation.png", dpi=150)
plt.show()

# =============================================================================
# FIGURE 3 — Vegetation overlay on a real test image
# Scans the full image pixel-by-pixel (striding by PATCH//2) and overlays
# a green mask wherever the model predicts vegetation.
# =============================================================================
def predict_image(model, img_path):
    """Return the RGB array and a boolean mask of model-predicted vegetation."""
    rgb_raw = np.array(Image.open(img_path)).astype(np.float32)
    rgb_norm = (rgb_raw - rgb_raw.mean()) / (rgb_raw.std() + 1e-8)
    H, W    = rgb_norm.shape[:2]
    half    = PATCH // 2
    stride  = PATCH // 2   # 50% overlap for smoother mask

    mask    = np.zeros((H, W), dtype=np.float32)
    counts  = np.zeros((H, W), dtype=np.float32)

    ys = range(half, H - half, stride)
    xs = range(half, W - half, stride)

    patches, coords = [], []
    for y in ys:
        for x in xs:
            patches.append(rgb_norm[y-half:y+half, x-half:x+half].transpose(2, 0, 1).ravel())
            coords.append((y, x))

    patches_t = torch.tensor(np.stack(patches))
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(patches_t), dim=1)[:, 1].numpy()  # veg probability

    for (y, x), p in zip(coords, probs):
        mask[y-half:y+half, x-half:x+half]   += p
        counts[y-half:y+half, x-half:x+half] += 1

    counts = np.maximum(counts, 1)
    return rgb_raw, mask / counts   # averaged probability per pixel


# Use the first test image for the visual
test_img_path = test_pairs[0][0]
rgb_raw, veg_prob = predict_image(model, test_img_path)

# Clip and normalise RGB for display
rgb_display = np.clip(rgb_raw / rgb_raw.max(), 0, 1)

# Build RGBA overlay: green where model is confident about vegetation
overlay = np.zeros((*veg_prob.shape, 4), dtype=np.float32)
overlay[..., 1] = 1.0          # green channel
overlay[..., 3] = np.clip((veg_prob - 0.5) * 2, 0, 1) * 0.6  # alpha from confidence

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Vegetation Detection — Test Image", fontsize=14)

axes[0].imshow(rgb_display)
axes[0].set_title("Original RGB Image")
axes[0].axis("off")

axes[1].imshow(veg_prob, cmap="RdYlGn", vmin=0, vmax=1)
axes[1].set_title("Model Vegetation Probability\n(green = high, red = low)")
axes[1].axis("off")
plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04, label="P(vegetation)")

axes[2].imshow(rgb_display)
axes[2].imshow(overlay)
axes[2].set_title("Overlay — Green = Predicted Vegetation\n(opacity = model confidence)")
axes[2].axis("off")
patch = mpatches.Patch(color=(0, 1, 0, 0.6), label="Predicted vegetation")
axes[2].legend(handles=[patch], loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("vegetation_overlay.png", dpi=150)
plt.show()

torch.save(model.state_dict(), "vegetation_nn.pt")
print("\nSaved: vegetation_nn.pt, learning_curves.png, test_evaluation.png, vegetation_overlay.png")