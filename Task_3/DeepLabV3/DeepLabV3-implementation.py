"""
Vegetation Segmentation on Sentinel-2 Satellite Imagery
=======================================================

Documentation written with assistance from Claude (Anthropic).

This script trains a model to look at satellite images and figure out which
parts are vegetation (trees, grass, crops) and which parts aren't (buildings,
water, roads, bare ground). It marks each pixel as either "vegetation" or
"not vegetation."

The model is DeepLabV3 with a ResNet-50 backbone, a well-established
architecture for this kind of pixel-by-pixel classification. It comes
pretrained on everyday photos, and we fine-tune it here for satellite data.

Architecture overview
---------------------
 
    Input image (4 bands: B, G, R, NIR)
                │
                ▼
    ┌───────────────────────┐
    │   ResNet-50 backbone  │   extracts visual features
    │   (pretrained)        │   (edges → shapes → patterns)
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │   ASPP module         │   looks at multiple scales at once,
    │   (multi-scale conv)  │   so tiny and large vegetation both work
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │   Classifier head     │   per-pixel prediction:
    │   (2 output classes)  │   vegetation or not
    └───────────────────────┘
                │
                ▼
    Output mask (same size as input)

"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Loading images and masks ─────────────────────────────────────────────────

def load_tiff(path, n_bands=4):
    """
    Load a satellite image from dataset and normalize it.

    Satellite images have varying pixel values depending on
    lighting, sensors, and what's being photographed. To make training
    stable, each band is flattened into the 0-1 range by clipping extreme
    outliers (top and bottom 1%) and rescaling the rest.

    If the file happens to have fewer than 4 bands, we fill in the missing
    ones with the average of what we have so the shape stays consistent.
    """
    with rasterio.open(path) as src:
        available = src.count
        k = min(n_bands, available)
        img = src.read(list(range(1, k + 1))).astype(np.float32)

    # Normalize each band independently to the 0-1 range.
    for c in range(img.shape[0]):
        lo, hi = np.percentile(img[c], (1, 99))
        img[c] = np.clip((img[c] - lo) / (hi - lo + 1e-6), 0, 1)

    # Pad missing bands if needed.
    if k < n_bands:
        pad = np.repeat(img.mean(axis=0, keepdims=True), n_bands - k, axis=0)
        img = np.concatenate([img, pad], axis=0)

    return img


def load_mask(path):
    """
    Load a vegetation mask. Masks are either stored as 0/1 or 0/255,
    so we handle both and return a clean 0-or-1 array.
    """
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.float32)
    if mask.max() > 1:
        mask = mask / 255.0
    return (mask > 0.5).astype(np.int64)


# ── Dataset ──────────────────────────────────────────────────────────────────

class VegetationDataset(Dataset):
    """
    Pairs up satellite images with their vegetation masks.

    Images in `samples_dir` and masks in `labels_dir` are matched by
    sorted filename order, so make sure corresponding files have matching
    names (e.g. tile_001.tif in both folders).

    Every image and mask gets resized to the same square size so they can
    be fed to the model in batches.
    """

    def __init__(self, samples_dir, labels_dir, img_size=256, n_bands=4):
        exts = ('.tif', '.tiff')
        self.samples = sorted(f for f in os.listdir(samples_dir) if f.lower().endswith(exts))
        self.labels  = sorted(f for f in os.listdir(labels_dir)  if f.lower().endswith(exts))
        self.samples_dir = samples_dir
        self.labels_dir  = labels_dir
        self.img_size = img_size
        self.n_bands  = n_bands
        assert len(self.samples) == len(self.labels) > 0, "samples/labels mismatch or empty"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img  = load_tiff(os.path.join(self.samples_dir, self.samples[idx]), self.n_bands)
        mask = load_mask(os.path.join(self.labels_dir,  self.labels[idx]))

        img  = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        # Resize both to the target size. Images use smooth interpolation;
        # masks use nearest-neighbor so we don't accidentally invent
        # in-between class values.
        img = F.interpolate(img.unsqueeze(0), size=self.img_size,
                            mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                             size=self.img_size, mode='nearest').squeeze().long()
        return img, mask


# ── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes=2, in_channels=4):
    """
    Set up the segmentation model.

    We start from a DeepLabV3 model that's already been trained on millions
    of regular photos, then adapt it for our task. Two changes are needed:

    First, the original model expects 3-channel (RGB) input, but we have 4
    channels (adding near-infrared). We swap in a new first layer that accepts
    4 channels, keeping the pretrained weights for the RGB channels and
    initializing the new NIR channel with the average of those weights. This
    way the model starts off already knowing how to see, rather than having
    to learn from scratch.

    Second, the original model was trained to recognize 21 different object
    types. We replace its final layer with one that outputs just 2 classes:
    vegetation or not.
    """
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    # Replace the first layer to accept 4 input channels instead of 3.
    if in_channels != 3:
        old = model.backbone.conv1
        new = nn.Conv2d(in_channels, old.out_channels,
                        kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            if in_channels > 3:
                new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        model.backbone.conv1 = new

    # New final layer that outputs our 2 classes.
    model.classifier = DeepLabHead(2048, num_classes)
    model.aux_classifier = None
    return model


# ── Training ─────────────────────────────────────────────────────────────────

    # epochs are set to a default of 20 if not assigned to the function call in main
def train(samples_dir, labels_dir, epochs=20, batch_size=4, lr=1e-4,
          img_size=256, n_bands=4):
    """
    Train the model.

    The dataset gets split 80/20 into training and validation. The model
    sees only the training portion during learning; the validation portion
    is used to check how well it generalizes. Whenever the model performs
    its best so far on validation, we save it.

    One detail worth knowing: errors are weighted on non-vegetation pixels
    more heavily (5x) than vegetation pixels. I tried the reverse approach 
    first and it resulted in a vegetation bias where the model would just 
    predict "vegetation" everywhere and return the image covered entirely 
    in a green overlay(detections). If a dataset is used that has the opposite 
    imbalance (more clouds than veg), flip the weights.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load and split the dataset. The fixed random seed means the same
    # files go into train/val every run, so results are comparable.
    # modify the seed for a more broad analysis of the dataset
    ds = VegetationDataset(samples_dir, labels_dir, img_size, n_bands)
    n_val = max(1, int(len(ds) * 0.2))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = build_model(num_classes=2, in_channels=n_bands).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_weights = torch.tensor([18.0, 1.0], device=device)  # [non-veg, veg]
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best = float('inf')

    for epoch in range(1, epochs + 1):
        # Training pass: show the model each batch and update its weights.
        model.train()
        tr = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)['out']
            loss = criterion(logits, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr += loss.item()

        # Validation pass: just measure performance, don't update anything.
        model.eval()
        va = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                va += criterion(model(imgs)['out'], masks).item()

        tr /= len(train_loader); va /= len(val_loader)
        print(f"Epoch {epoch:>2}/{epochs}  train {tr:.4f}  val {va:.4f}")

        # Save the model if it's the best one we've seen so far.
        if va < best:
            best = va
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ saved (val {best:.4f})")

    print(f"\nDone. Best val loss: {best:.4f}")


# ── Visualizing results ──────────────────────────────────────────────────────

def visualize(samples_dir, labels_dir, img_size=256, n_bands=4, max_samples=8):
    """
    Generate side-by-side result images for a handful of validation samples.

    Each result image has three panels:
        - The original satellite image in RGB.
        - A heatmap showing how confident the model is that each pixel
          is vegetation (green = confident yes, red = confident no).
        - The original image with a green overlay marking predicted
          vegetation. More transparent areas mean the model is less sure.

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Recreate the exact same val split used during training.
    ds = VegetationDataset(samples_dir, labels_dir, img_size, n_bands)
    n_val = max(1, int(len(ds) * 0.2))
    _, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                             generator=torch.Generator().manual_seed(42))

    model = build_model(num_classes=2, in_channels=n_bands).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    veg_patch = mpatches.Patch(color='limegreen', alpha=0.85, label='Predicted vegetation')

    with torch.no_grad():
        for i in range(min(max_samples, len(val_ds))):
            img, _ = val_ds[i]

            # Run the image through the model and pull out the vegetation
            # probability for each pixel.
            logits = model(img.unsqueeze(0).to(device))['out']
            prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

            # Build a displayable RGB picture from the R, G, and B bands.
            # (The bands are stored in B, G, R, NIR order, so we reorder.)
            rgb = np.stack([img[2].numpy(), img[1].numpy(), img[0].numpy()], axis=-1)
            rgb = np.clip(rgb, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(f"Vegetation Detection — Sample {i + 1}")

            axes[0].imshow(rgb)
            axes[0].set_title("Original RGB Image")
            axes[0].axis('off')

            im = axes[1].imshow(prob, cmap='RdYlGn', vmin=0, vmax=1)
            axes[1].set_title("Model Vegetation Probability\n(green = high, red = low)")
            axes[1].axis('off')
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='P(vegetation)')

            # Green overlay where the model thinks there's vegetation.
            # The more confident it is, the more opaque the green.
            axes[2].imshow(rgb)
            overlay = np.zeros((*prob.shape, 4), dtype=np.float32)
            overlay[..., 1] = 0.9
            overlay[..., 3] = np.where(prob > 0.5, prob * 0.75, 0)
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay — Green = Predicted Vegetation\n(opacity = model confidence)")
            axes[2].axis('off')
            axes[2].legend(handles=[veg_patch], loc='lower right', fontsize=8)

            plt.tight_layout()
            plt.savefig(f"result_{i + 1}.png", dpi=120, bbox_inches='tight')
            plt.show(); plt.close()

    print(f"Saved {min(max_samples, len(val_ds))} figures.")


# ── Run it ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLES_DIR = ".../samples" # add your own path here
    LABELS_DIR  = ".../labels"  # add your own path here

    train(SAMPLES_DIR, LABELS_DIR, epochs=10, batch_size=4, lr=1e-4, img_size=256, n_bands=4)
    visualize(SAMPLES_DIR, LABELS_DIR, img_size=256, n_bands=4, max_samples=8)