"""
=============================================================================
VEGETATION CLASSIFICATION — DEEP LEARNING PIPELINE (PyTorch)
=============================================================================
REFERENCE FILE: Keep this as a template for future deep learning projects.

TASK:
    Given paired satellite images (.tiff):
      - "samples/"  → RGB images of terrain         (e.g. ...SAFE_img_0.tiff)
      - "labels/"   → Grayscale NDVI masks           (e.g. ...SAFE_ndvi_0.tiff)
    Train a CNN to predict vegetation type per pixel (semantic segmentation).

ARCHITECTURE OVERVIEW:
    ┌─────────────────────────────────────────────────────────────┐
    │  VegetationDataset  →  pairs + loads the .tiff files        │
    │  Augmentation       →  random flips/rotations on image+mask │
    │  Preprocessing      →  normalizes pixel values              │
    │  VegetationModel    →  Encoder → Bottleneck → Decoder CNN   │
    │  Trainer            →  explicit training loop (PyTorch way) │
    │  Evaluator          →  accuracy, precision, recall, F1      │
    │  main()             →  orchestrates everything              │
    └─────────────────────────────────────────────────────────────┘

DEPENDENCIES:
    pip install torch torchvision tifffile numpy pillow scikit-learn matplotlib time
=============================================================================

Done by Claude
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os           # for navigating the file system (listing folders, joining paths)
import re           # regular expressions — used to extract the index from filenames
import random       # Python's built-in random, used for reproducibility seeds
import time         # for measuring how long training takes

import torch                                # the core PyTorch library
import torch.nn as nn                       # neural network building blocks (Conv2d, ReLU, etc.)
import torch.nn.functional as F             # functional versions of layers (used inside forward())
import torch.optim as optim                 # optimizers (Adam, SGD, etc.)
from torch.utils.data import (
    Dataset,                                # base class — every custom dataset extends this
    DataLoader,                             # wraps a Dataset and handles batching + shuffling
    random_split                            # splits a dataset into train/val/test portions
)
import torchvision.transforms.functional as TF  # functional image transforms (no randomness built-in,
                                                 # so WE control when/how they apply — important for masks)

import numpy as np                          # numerical operations on arrays
from PIL import Image                       # Pillow — used to load .tiff image files
import tifffile                             # better .tiff support for scientific imagery (float32, 16-bit, etc.)
import matplotlib.pyplot as plt             # for visualizing images and training curves

from sklearn.metrics import (               # scikit-learn metrics — same library used for SVM evaluation
    accuracy_score,                         # overall correctness
    precision_score,                        # how precise positive predictions are
    recall_score,                           # how many real positives we caught
    f1_score,                               # harmonic mean of precision and recall
    classification_report,                  # prints a full breakdown per class
    confusion_matrix                        # NxN matrix of true vs predicted class counts
)


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
# Centralising all hyperparameters here means you never have to hunt through
# the code to change a learning rate or batch size. Change it once, here.

CONFIG = {
    # --- Paths ---
    "samples_dir"     : "samples/",
    "labels_dir"      : "labels/",
 
    # --- Image settings ---
    "image_size"      : (128, 128),         # (H, W) — TODO: tune for your GPU memory
 
    # --- Label / class settings ---
    # NDVI physical range: -1.0 to +1.0
    #   < 0.0       → class 0  Non-vegetal      (water, rock, bare soil)
    #   0.0 – 0.3   → class 1  Low vegetation   (sparse grass, shrubs)
    #   0.3 – 0.6   → class 2  Moderate veg.    (farmland, mixed cover)
    #   > 0.6       → class 3  Dense vegetation (forest, healthy crops)
    "num_classes"     : 4,
    "ndvi_thresholds" : [0.0, 0.3, 0.6],   # cut-points on the [-1, 1] scale
 
    # --- Data split ---
    "train_ratio"     : 0.70,
    "val_ratio"       : 0.15,
    "test_ratio"      : 0.15,
 
    # --- Training hyperparameters ---
    "batch_size"      : 4,
    "epochs"          : 50,
    "learning_rate"   : 1e-4,
    "weight_decay"    : 1e-5,
 
    # ─────────────────────────────────────────────────────────────────────────
    # AUGMENTATION PARAMETERS
    # ─────────────────────────────────────────────────────────────────────────
    # Each "prob" key is the probability (0–1) that the transform fires on any
    # single image. 0.5 = fires half the time. These are all independent —
    # a single image may receive any combination of transforms in one pass.
    #
    # WHY KEEP PROBABILITIES MODERATE (0.3–0.5)?
    # If every transform fires every time (prob=1.0), the model only ever sees
    # heavily distorted images and may struggle to converge. Moderate probabilities
    # ensure it also sees "clean" examples, which stabilises training.
    "use_augmentation"  : True,             # master switch — False disables all augmentation
 
    # 1. Flips
    "flip_prob"         : 0.5,             # horizontal flip — valid for top-down satellite views
    "vflip_prob"        : 0.3,             # vertical flip
 
    # 2. Rotation
    "rotate_prob"       : 0.4,
    "rotate_range"      : 30,              # degrees — rotates between -30° and +30°
                                            # Beyond ±45° starts to look unrealistic for aerial imagery
 
    # 3. Brightness & Contrast (colour transforms — image only, never mask)
    "brightness_prob"   : 0.3,
    "brightness_range"  : (0.7, 1.3),      # factor: 0.7=darker, 1.3=brighter
 
    "contrast_prob"     : 0.3,             # contrast: how spread out light/dark values are
    "contrast_range"    : (0.75, 1.25),    # factor < 1 = flatter image, > 1 = sharper
                                            # Tighter than brightness: ±25% is already visible.
                                            # Simulates haze, atmospheric scattering, sensor variation.
 
    # 4. Scale (zoom in/out — spatial transform → image AND mask)
    "scale_prob"        : 0.3,
    "scale_range"       : (0.85, 1.15),    # < 1.0 = zoom out (pad edges), > 1.0 = zoom in (crop)
                                            # Kept within ±15% so features remain identifiable.
                                            # Beyond ±30% cuts off too much of the scene.
 
    # 5. Translation (shift — spatial transform → image AND mask)
    "translate_prob"    : 0.3,
    "translate_range"   : 0.10,            # fraction of image size: 0.10 = up to ±10% shift
                                            # A 128×128 image shifts by at most ±12 pixels.
                                            # Beyond 20% causes too much black padding / info loss.
 
    # --- Reproducibility ---
    "seed"            : 42,
 
    # --- Output ---
    "model_save_path" : "vegetation_model.pth",
    "device": (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ),
}


# =============================================================================
# UTILITY — SET RANDOM SEEDS
# =============================================================================

def set_seed(seed: int):
    """
    Force all random number generators to use the same seed.

    Without this, two runs of the same code produce different results
    (different train/val splits, different augmentations, etc.).

    Why multiple seeds? Because Python, NumPy, and PyTorch each have their
    own internal random generators — you need to seed all of them.
    """
    random.seed(seed)           # Python's built-in random module
    np.random.seed(seed)        # NumPy's random module
    torch.manual_seed(seed)     # PyTorch CPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU operations


# =============================================================================
# CLASS 1: VegetationDataset
# =============================================================================
# In PyTorch, every dataset must be a class that extends torch.utils.data.Dataset.
# The two methods you MUST implement are:
#   __len__  → returns how many samples exist
#   __getitem__ → returns one (image, label) pair by index
#
# The DataLoader calls these automatically when building batches.

class VegetationDataset(Dataset):
    """
    Loads paired RGB images and NDVI label masks from disk.

    Pairing logic:
        Both filenames share the same satellite scene prefix and trailing
        index, but differ only in the middle tag ('img' vs 'ndvi'). E.g.:

            samples/ ...T19QGA_20230910T203157.SAFE_img_99.tiff
            labels/  ...T19QGA_20230910T203157.SAFE_ndvi_99.tiff

        Crucially the index RESETS to 0 for each new scene, so the plain
        number alone is not unique. We build the pairing key from the FULL
        prefix (everything before _img_ / _ndvi_) PLUS the number. That
        combination is unique across the entire dataset.

    What this class does NOT do:
        - Augmentation (handled by Augmentation class)
        - Normalisation of RGB images (handled by Preprocessing class)
        These are kept separate so they can be swapped independently.
    """

    # Regex that captures:
    #   group 1 → everything before _img_ or _ndvi_   (the scene prefix)
    #   group 2 → img | ndvi                           (the file type tag)
    #   group 3 → the trailing number                  (the patch index)
    # Example: "...SAFE_img_99.tiff"
    #   → prefix = "...SAFE", tag = "img", number = "99"
    _FILENAME_RE = re.compile(r'^(.+)_(img|ndvi)_(\d+)\.tiff$')

    def __init__(self, samples_dir: str, labels_dir: str,
                 preprocessing=None, augmentation=None,
                 image_size=(256, 256), ndvi_thresholds=(0.0, 0.3, 0.6)):
        """
        Args:
            samples_dir      : path to folder of RGB .tiff images
            labels_dir       : path to folder of NDVI .tiff masks
            preprocessing    : a Preprocessing instance (or None)
            augmentation     : an Augmentation instance (or None)
            image_size       : (H, W) to resize every image to
            ndvi_thresholds  : cut-points on the [-1, 1] NDVI scale that
                               separate the vegetation classes
        """
        self.samples_dir     = samples_dir
        self.labels_dir      = labels_dir
        self.preprocessing   = preprocessing
        self.augmentation    = augmentation
        self.image_size      = image_size
        self.ndvi_thresholds = ndvi_thresholds

        # Build the paired list of file paths
        self.pairs = self._build_pairs()

        # Compute the global min/max of the raw label values so we can
        # normalise every label to [-1, 1] consistently. We scan all label
        # files once here, at dataset creation time, and reuse the result
        # for every __getitem__ call.
        print("[Dataset] Computing global label statistics for NDVI normalisation...")
        self.global_min, self.global_max = self._compute_label_stats()
        print(f"[Dataset] Label raw range → min: {self.global_min:.4f}  max: {self.global_max:.4f}")
        print(f"[Dataset] Found {len(self.pairs)} image–label pairs.")

    # -------------------------------------------------------------------------
    def _extract_pairing_key(self, filename: str):
        """
        Extracts the unique pairing key from a filename.

        The key is (prefix, number) — a tuple that is guaranteed to be
        unique even when the trailing number resets across scenes.

        Example:
            "S2A_...T19QGA_20230910T203157.SAFE_img_99.tiff"
                → key = ("S2A_...T19QGA_20230910T203157.SAFE", 99)

            "S2A_...T19QFA_20230809T224110.SAFE_ndvi_0.tiff"
                → key = ("S2A_...T19QFA_20230809T224110.SAFE", 0)

        Those two are different keys (different prefix), so they will NOT
        be paired with each other — which is exactly what we want.

        Returns None if the filename doesn't match the expected pattern
        (so we can skip unexpected files silently).
        """
        match = self._FILENAME_RE.match(filename)
        if match is None:
            return None                     # filename doesn't fit our convention — skip it
        prefix = match.group(1)            # the full scene identifier
        number = int(match.group(3))       # the patch index within that scene
        return (prefix, number)            # unique composite key

    # -------------------------------------------------------------------------
    def _build_pairs(self) -> list:
        """
        Scans both directories and pairs files by their composite key.

        Returns a sorted list of tuples: [(sample_path, label_path), ...]
        Sorted by key so the order is deterministic across runs.
        """
        # List only .tiff files (ignore hidden files like .DS_Store)
        sample_files = [f for f in os.listdir(self.samples_dir) if f.endswith('.tiff')]
        label_files  = [f for f in os.listdir(self.labels_dir)  if f.endswith('.tiff')]

        # Build dictionaries: {(prefix, number): full_path}
        # Files that don't match the regex are silently dropped (key = None).
        sample_map = {}
        for f in sample_files:
            key = self._extract_pairing_key(f)
            if key is not None:
                sample_map[key] = os.path.join(self.samples_dir, f)

        label_map = {}
        for f in label_files:
            key = self._extract_pairing_key(f)
            if key is not None:
                label_map[key] = os.path.join(self.labels_dir, f)

        # Keep only keys that appear in BOTH folders.
        # sorted() on tuples sorts first by prefix (alphabetical = chronological
        # for Sentinel-2 scene IDs), then by number within each scene.
        shared_keys = sorted(set(sample_map.keys()) & set(label_map.keys()))

        if len(shared_keys) == 0:
            raise ValueError(
                "[Dataset] No matching pairs found.\n"
                "  Check that filenames follow the pattern: ...SAFE_img_N.tiff / ...SAFE_ndvi_N.tiff"
            )

        return [(sample_map[k], label_map[k]) for k in shared_keys]

    # -------------------------------------------------------------------------
    def _compute_label_stats(self):
        """
        Scans ALL label files to find the global minimum and maximum raw value.

        This is needed by normalize_to_ndvi(): the normalisation formula
        uses the global range so that every file is scaled consistently.
        Without global stats, two files with different local ranges would
        produce different NDVI values for the same physical measurement.

        Returns:
            global_min : float — lowest raw value across all label files
            global_max : float — highest raw value across all label files
        """
        global_min =  float('inf')
        global_max = -float('inf')

        for _, label_path in self.pairs:
            raw = tifffile.imread(label_path).astype(np.float32)
            global_min = min(global_min, float(raw.min()))
            global_max = max(global_max, float(raw.max()))

        return global_min, global_max

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_to_ndvi(label: np.ndarray,
                          global_min: float,
                          global_max: float) -> np.ndarray:
        """
        Rescales raw label pixel values from their instrument range into
        the standard NDVI range of [-1.0, +1.0].

        WHY IS THIS NEEDED?
        -------------------
        NDVI is physically defined as:
            NDVI = (NIR - Red) / (NIR + Red)
        which always lies in [-1, 1]. But the .tiff files store the raw
        instrument counts (e.g., uint16 in [0, 65535] or float32 in
        [0, 10000]) — NOT the final NDVI value. We need to map that raw
        range back to [-1, 1] to recover the physical meaning.

        HOW IT WORKS:
        -------------
        The formula is a two-step linear transform:

          Step 1 — map [global_min, global_max] → [0, 1]:
              t = (label - global_min) / (global_max - global_min)

          Step 2 — map [0, 1] → [-1, 1]:
              ndvi = t * 2 - 1

        Combined:
              ndvi = (label - global_min) / (global_max - global_min) * 2 - 1

        Example:
            raw value = global_min  → NDVI = -1.0  (bare rock, deep water)
            raw value = midpoint    → NDVI =  0.0  (neutral / sparse cover)
            raw value = global_max  → NDVI = +1.0  (dense healthy vegetation)

        Args:
            label      : (H, W) or (H, W, 1) float32 array of raw values
            global_min : minimum raw value across ALL files in the dataset
            global_max : maximum raw value across ALL files in the dataset

        Returns:
            ndvi : same shape as input, float32, values in [-1.0, 1.0]
        """
        label = label.astype(np.float32)
        ndvi  = (label - global_min) / (global_max - global_min) * 2 - 1
        return ndvi

    # -------------------------------------------------------------------------
    def _load_label_raw(self, path: str) -> np.ndarray:
        """
        Loads a label .tiff file and returns it as a FLOAT32 array.

        Unlike _load_image() (which converts everything to uint8 for display),
        labels MUST keep their original float precision so that
        normalize_to_ndvi() can do an accurate linear rescale to [-1, 1].
        Converting to uint8 first would destroy the sub-integer precision
        needed to distinguish NDVI = 0.28 from NDVI = 0.31, for example.

        Returns:
            raw : (H, W) float32 array — raw instrument values
        """
        raw = tifffile.imread(path).astype(np.float32)

        # Squeeze any extra dimensions so we always get a plain (H, W) array
        if raw.ndim == 3:
            raw = raw[:, :, 0] if raw.shape[2] == 1 else raw.mean(axis=2)
        if raw.ndim == 4:
            raw = raw[0, :, :, 0]   # edge case: some tiff writers add a batch dim

        return raw  # shape: (H, W), dtype: float32

    # -------------------------------------------------------------------------
    def _ndvi_to_class(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Converts a [-1, 1] NDVI array into integer class indices.

        Thresholds from CONFIG (default: [0.0, 0.3, 0.6]):

            NDVI < 0.0          → class 0  (Non-vegetal:       water, rock, bare soil)
            0.0 ≤ NDVI < 0.3    → class 1  (Low vegetation:    sparse grass, shrubs)
            0.3 ≤ NDVI < 0.6    → class 2  (Moderate veg.:     farmland, mixed cover)
            NDVI ≥ 0.6          → class 3  (Dense vegetation:  forest, healthy crops)

        The logic works by starting every pixel at class 0 and incrementing
        it each time its NDVI value crosses a threshold — identical to the
        SVM classification used in Task 2.

        Args:
            ndvi : (H, W) float32 array of NDVI values in [-1, 1]

        Returns:
            class_mask : (H, W) int64 array of class indices {0, 1, 2, 3}
        """
        # Everything starts at class 0 (the "below all thresholds" bin)
        class_mask = np.zeros(ndvi.shape[:2], dtype=np.int64)

        for class_id, threshold in enumerate(self.ndvi_thresholds):
            # Pixels ABOVE this threshold belong to the next class or higher.
            # By iterating in order, each assignment overwrites the previous,
            # so only the highest applicable class survives.
            class_mask[ndvi > threshold] = class_id + 1

        return class_mask  # shape: (H, W), values: 0, 1, 2, 3

    # -------------------------------------------------------------------------
    def _load_image(self, path: str) -> np.ndarray:
        """
        Loads a RGB .tiff file and returns it as a uint8 numpy array (H, W, C).

        tifffile handles edge cases that PIL sometimes struggles with
        (float32 satellite imagery, 16-bit depth, etc.).

        WHY DO WE CONVERT TO uint8?
        ----------------------------
        PIL's Image.fromarray() — which we use later for augmentation and
        resizing — only accepts a small set of array types. uint8 (values
        0–255) is the universal format it handles cleanly. So the conversion
        is a compatibility step, NOT a loss of information for the RGB image:
        standard display-ready RGB is already 8-bit per channel.

        WHAT IF THE FILE IS NOT uint8?
        --------------------------------
        Satellite imagery often ships in other formats, for example:

          • float32 in [0.0, 1.0]   — some sensors store reflectance directly
          • uint16 in [0, 10 000]   — Sentinel-2 stores surface reflectance ×10 000
          • uint16 in [0, 65 535]   — full 16-bit range

        We handle all of these with a single min-max rescale:

            scaled = (value - file_min) / (file_max - file_min)   → [0.0, 1.0]
            uint8  = scaled × 255                                  → [0, 255]

        This stretches whatever range the file uses to fill the 0–255 display
        range. For RGB images that's fine — we only care about relative colour
        values, not the absolute physical magnitude.

        EDGE CASE — a completely uniform image (all pixels the same value):
            file_min == file_max, so the denominator would be zero → division error.
            We catch this and return a black image (all zeros) instead, because
            a uniform image carries no useful information anyway.

        NOTE: We do NOT use this function for label files. Labels store NDVI
        values where the absolute magnitude matters (0.28 ≠ 0.31). Converting
        labels to uint8 would round away that precision. Labels are loaded
        through _load_label_raw() which keeps them as float32.
        """
        img = tifffile.imread(path)  # shape could be (H, W), (H, W, C), or (C, H, W)

        # Ensure we have a channel dimension even for grayscale images
        if img.ndim == 2:
            img = img[:, :, np.newaxis]  # (H, W) → (H, W, 1)

        # If channels are first (C, H, W), move them to last (H, W, C)
        if img.shape[0] <= 4 and img.ndim == 3 and img.shape[0] < img.shape[1]:
            img = np.transpose(img, (1, 2, 0))  # (C, H, W) → (H, W, C)

        # Convert to uint8 if the file is in any other numeric format
        if img.dtype != np.uint8:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                # Linear stretch: map [img_min, img_max] → [0, 255]
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                # Uniform image — just return black
                img = np.zeros_like(img, dtype=np.uint8)

        return img

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        """PyTorch requires this. DataLoader calls it to know the dataset size."""
        return len(self.pairs)

    # -------------------------------------------------------------------------
    def __getitem__(self, idx: int):
        """
        Returns one (image_tensor, label_tensor) pair.

        PyTorch's DataLoader calls this repeatedly to build each batch.
        The index `idx` is chosen by the DataLoader (randomly if shuffle=True).

        Returns:
            image  : FloatTensor of shape (C, H, W)  e.g. (3, 256, 256)
            label  : LongTensor  of shape (H, W)     e.g. (256, 256) — one class per pixel
        """
        sample_path, label_path = self.pairs[idx]

        # --- Load RGB image → uint8 array ---
        image_np = self._load_image(sample_path)   # (H, W, 3) uint8

        # --- Load label as raw float, then normalise to [-1, 1] NDVI ---
        label_raw  = self._load_label_raw(label_path)                        # (H, W) float32
        label_ndvi = self.normalize_to_ndvi(label_raw,
                                            self.global_min,
                                            self.global_max)                 # (H, W) float32 in [-1, 1]

        # --- Resize image to target size ---
        image_pil = Image.fromarray(image_np).resize(
            (self.image_size[1], self.image_size[0]),  # PIL uses (W, H), not (H, W)
            Image.BILINEAR                              # smooth interpolation for the RGB image
        )

        # --- Resize NDVI map using NEAREST neighbour ---
        # We convert the float NDVI to a PIL image for resizing. PIL doesn't
        # support float32 directly, so we temporarily scale to uint16 range,
        # resize with NEAREST (to avoid inventing fake NDVI values at boundaries),
        # then scale back to float32. NEAREST is mandatory here: any blending
        # between neighbour pixels would produce NDVI values that never existed
        # in the original data, corrupting the class boundaries.
        ndvi_uint16  = ((label_ndvi + 1) / 2 * 65535).astype(np.uint16)  # [-1,1] → [0, 65535]
        ndvi_pil     = Image.fromarray(ndvi_uint16).resize(
            (self.image_size[1], self.image_size[0]),
            Image.NEAREST
        )
        ndvi_resized = np.array(ndvi_pil).astype(np.float32) / 65535 * 2 - 1  # back to [-1, 1]

        # --- Convert NDVI [-1, 1] → class indices {0, 1, 2, 3} ---
        label_class = self._ndvi_to_class(ndvi_resized)  # (H, W) int64

        # --- Apply augmentation (if provided) ---
        # IMPORTANT: augmentation must be applied to image AND mask with the
        # same random parameters. That's why we pass both to the Augmentation class.
        if self.augmentation is not None:
            image_pil, label_class = self.augmentation(image_pil, label_class)

        # --- Convert image to tensor and apply normalisation ---
        # TF.to_tensor converts a PIL image (H, W, C) uint8 → FloatTensor (C, H, W) in [0, 1]
        image_tensor = TF.to_tensor(image_pil)  # shape: (3, H, W), values: 0.0–1.0

        if self.preprocessing is not None:
            image_tensor = self.preprocessing(image_tensor)

        # --- Convert label to tensor ---
        # Labels need to be LongTensor (int64) — that's what CrossEntropyLoss expects
        label_tensor = torch.from_numpy(label_class).long()  # shape: (H, W)

        return image_tensor, label_tensor


# =============================================================================
# CLASS 2: Augmentation
# =============================================================================
#
# GROUND RULE FOR SEGMENTATION AUGMENTATION:
#
#   Spatial transforms  (rotation, flip, scale, translate)
#       → apply IDENTICALLY to image AND mask
#       → the same random parameter is drawn ONCE and used for both
#       → reason: the mask is a spatial map; any positional shift in the
#         image must be mirrored in the mask or labels point at wrong pixels
#
#   Colour transforms   (brightness, contrast)
#       → apply to IMAGE ONLY — never to the mask
#       → reason: the mask stores integer class indices (0/1/2/3), not colour
#         values. Adjusting contrast on a mask would produce fractional indices
#         that map to nonexistent classes.
#
# WHY NOT USE torchvision.transforms.Compose?
#   Compose applies each transform to the image independently, with its own
#   internal random state. We cannot guarantee the mask receives the same
#   spatial parameters. We therefore draw random values ourselves and apply
#   manually to both inputs.
 
class Augmentation:
    """
    Applies the four required augmentation types to an (image, mask) pair.
 
    Implemented transforms:
        1. Horizontal / vertical flip   (spatial)
        2. Random rotation ±30°         (spatial)
        3. Brightness + contrast        (colour, image only)
        4. Scale (zoom) + translate     (spatial)
    """
 
    def __init__(self,
                 flip_prob        = CONFIG["flip_prob"],
                 vflip_prob       = CONFIG["vflip_prob"],
                 rotate_prob      = CONFIG["rotate_prob"],
                 rotate_range     = CONFIG["rotate_range"],
                 brightness_prob  = CONFIG["brightness_prob"],
                 brightness_range = CONFIG["brightness_range"],
                 contrast_prob    = CONFIG["contrast_prob"],
                 contrast_range   = CONFIG["contrast_range"],
                 scale_prob       = CONFIG["scale_prob"],
                 scale_range      = CONFIG["scale_range"],
                 translate_prob   = CONFIG["translate_prob"],
                 translate_range  = CONFIG["translate_range"]):
 
        self.flip_prob        = flip_prob
        self.vflip_prob       = vflip_prob
        self.rotate_prob      = rotate_prob
        self.rotate_range     = rotate_range
        self.brightness_prob  = brightness_prob
        self.brightness_range = brightness_range
        self.contrast_prob    = contrast_prob
        self.contrast_range   = contrast_range
        self.scale_prob       = scale_prob
        self.scale_range      = scale_range
        self.translate_prob   = translate_prob
        self.translate_range  = translate_range
 
    # -------------------------------------------------------------------------
    def __call__(self, image: Image.Image, mask: np.ndarray):
        """
        Apply all transforms in sequence to (image, mask).
 
        Convention: spatial transforms first, colour transforms last.
        This keeps the pipeline easy to reason about and matches standard
        practice in segmentation data pipelines.
        """
 
        # ── 1. HORIZONTAL FLIP ───────────────────────────────────────────────
        # Mirror left ↔ right. Valid for top-down satellite views — there is
        # no "correct" horizontal orientation when looking straight down.
        # SPATIAL → both image and mask.
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask  = np.fliplr(mask).copy()
            # .copy() removes the negative memory stride numpy creates when
            # flipping in-place, which would cause torch.from_numpy() to crash.
 
        # ── 2. VERTICAL FLIP ─────────────────────────────────────────────────
        # Mirror top ↔ bottom. Also valid for overhead imagery.
        # SPATIAL → both.
        if random.random() < self.vflip_prob:
            image = TF.vflip(image)
            mask  = np.flipud(mask).copy()
 
        # ── 3. RANDOM ROTATION ───────────────────────────────────────────────
        # ONE angle is drawn and applied to BOTH. This is the critical step —
        # if image and mask rotated by different angles, labels misalign.
        #
        # Border pixels (corners exposed by rotation) are filled with 0.
        # For the image: black (0,0,0). For the mask: class 0 (non-vegetal).
        # These border pixels carry no real information so class 0 is acceptable.
        #
        # Image uses BILINEAR (smooth). Mask uses NEAREST to prevent fractional
        # class indices being created at the boundary between two classes.
        # SPATIAL → both.
        if random.random() < self.rotate_prob:
            angle    = random.uniform(-self.rotate_range, self.rotate_range)
            image    = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask_pil = Image.fromarray(mask.astype(np.int32))
            mask_pil = TF.rotate(mask_pil, angle, interpolation=TF.InterpolationMode.NEAREST)
            mask     = np.array(mask_pil).astype(np.int64)
 
        # ── 4. SCALING (random zoom in / out) ────────────────────────────────
        # Simulates viewing the same patch at a slightly different altitude or
        # zoom level. A scale < 1.0 zooms out (shrinks scene, pads edges with
        # black). A scale > 1.0 zooms in (magnifies scene, crops edges).
        #
        # In both cases the output is resized back to the original dimensions
        # so the tensor shape fed to the model stays constant.
        #
        # SPATIAL → both image and mask.
        if random.random() < self.scale_prob:
            scale    = random.uniform(*self.scale_range)
            orig_w, orig_h = image.size        # PIL size format: (width, height)
            new_w    = int(orig_w * scale)
            new_h    = int(orig_h * scale)
 
            img_s    = image.resize((new_w, new_h), Image.BILINEAR)
            msk_s    = Image.fromarray(mask.astype(np.int32)).resize(
                           (new_w, new_h), Image.NEAREST)
 
            if scale < 1.0:
                # Zoom OUT: scaled image is smaller → paste it centred on a
                # black canvas of the original size.
                pl = (orig_w - new_w) // 2    # left padding
                pt = (orig_h - new_h) // 2    # top  padding
                canvas_i = Image.new("RGB", (orig_w, orig_h), (0, 0, 0))
                canvas_m = Image.new("I",   (orig_w, orig_h), 0)   # "I" = 32-bit int
                canvas_i.paste(img_s, (pl, pt))
                canvas_m.paste(msk_s, (pl, pt))
                image = canvas_i
                mask  = np.array(canvas_m).astype(np.int64)
            else:
                # Zoom IN: scaled image is larger → crop the centre back down.
                cl = (new_w - orig_w) // 2    # left crop start
                ct = (new_h - orig_h) // 2    # top  crop start
                image = img_s.crop((cl, ct, cl + orig_w, ct + orig_h))
                mask  = np.array(msk_s)[ct:ct + orig_h, cl:cl + orig_w].astype(np.int64)
 
        # ── 5. TRANSLATION (random horizontal / vertical shift) ───────────────
        # Shifts the scene by a small number of pixels in any direction.
        # Simulates extracting the same patch at a slightly different position
        # within the original satellite tile.
        #
        # translate_range is a fraction of the image dimension (0.10 = 10%).
        # Pixels shifted out of frame are replaced with black / class 0.
        # We use PIL's affine transform with a pure translation matrix.
        #
        # PIL affine convention: the matrix specifies where each OUTPUT pixel
        # comes FROM in the input (inverse mapping). So to shift the image
        # by +dx pixels to the right, we pass -dx in the matrix.
        #
        # SPATIAL → both image and mask.
        if random.random() < self.translate_prob:
            w, h   = image.size
            max_dx = int(w * self.translate_range)
            max_dy = int(h * self.translate_range)
            dx     = random.randint(-max_dx, max_dx)
            dy     = random.randint(-max_dy, max_dy)
 
            affine = (1, 0, -dx, 0, 1, -dy)   # (a, b, c, d, e, f) — pure translation
 
            image    = image.transform(image.size, Image.AFFINE, affine,
                                       resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            mask_pil = Image.fromarray(mask.astype(np.int32))
            mask_pil = mask_pil.transform(mask_pil.size, Image.AFFINE, affine,
                                          resample=Image.NEAREST, fillcolor=0)
            mask     = np.array(mask_pil).astype(np.int64)
 
        # ── 6. BRIGHTNESS ADJUSTMENT — image only ────────────────────────────
        # Multiplies every pixel by a random factor. Simulates different times
        # of day, seasonal lighting variation, or sensor gain differences.
        # COLOUR → image only.
        if random.random() < self.brightness_prob:
            image = TF.adjust_brightness(image, random.uniform(*self.brightness_range))
 
        # ── 7. CONTRAST ADJUSTMENT — image only ──────────────────────────────
        # Contrast = the spread between dark and bright pixels.
        # adjust_contrast moves each pixel toward (factor < 1) or away from
        # (factor > 1) the image mean:
        #   factor < 1.0 → flatter image (pixels cluster toward grey)
        #   factor > 1.0 → sharper image (pixels spread further from grey)
        #
        # Simulates haze, thin cloud cover, atmospheric scattering, or
        # variation between satellite passes.
        # COLOUR → image only.
        if random.random() < self.contrast_prob:
            image = TF.adjust_contrast(image, random.uniform(*self.contrast_range))
 
        return image, mask


# =============================================================================
# CLASS 3: Preprocessing
# =============================================================================
# After loading and augmenting, we normalise the image tensor.
# Normalisation means shifting pixel values so they have:
#   mean ≈ 0  and  std ≈ 1
# This keeps the numbers fed into the network in a "comfortable" range,
# which makes training faster and more stable.
#
# The mean and std values below are computed from the ImageNet dataset
# (millions of natural images). Using these is standard practice even for
# satellite imagery when using pretrained backbones — the network was already
# trained expecting these statistics, so we match them.

class Preprocessing:
    """
    Normalises an image tensor using channel-wise mean and standard deviation.

    Input:  FloatTensor (C, H, W) with values in [0, 1]   ← from TF.to_tensor()
    Output: FloatTensor (C, H, W) with values roughly in [-2, 2]

    Formula applied per channel:
        normalised = (pixel_value - mean) / std
    """

    # ImageNet statistics — used as a standard starting point
    # Each value corresponds to one RGB channel (R, G, B)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, mean=None, std=None):
        """
        Args:
            mean : list of per-channel means. Defaults to ImageNet means.
            std  : list of per-channel stds.  Defaults to ImageNet stds.

        TODO: For best results, compute mean and std from YOUR dataset instead.
              Run a pass over all training images and compute per-channel statistics.
              This is called "dataset normalisation" and gives a small but real improvement.
        """
        self.mean = mean if mean is not None else self.IMAGENET_MEAN
        self.std  = std  if std  is not None else self.IMAGENET_STD

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies channel-wise normalisation to a (C, H, W) tensor.
        TF.normalize does this efficiently in one call.
        """
        return TF.normalize(tensor, mean=self.mean, std=self.std)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverses normalisation — useful when visualising images after training.
        You need this because the raw normalised tensor looks washed out.
        """
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std  = torch.tensor(self.std).view(3, 1, 1)
        return tensor * std + mean  # invert: pixel = normalised * std + mean


# =============================================================================
# CLASS 4: CNN model
# =============================================================================

# TODO: Create and fine-tune cnn model

# =============================================================================
# CLASS 4b: AugmentationInspector
# =============================================================================
# Augmentation does NOT change the model — it changes the DATA fed into it.
# The parameter count is therefore identical whether augmentation is on or off.
#
# To confirm augmentation is actually working, we need to LOOK at the data
# directly. This class provides two types of visual evidence:
#
#   1. show_augmentation_grid()
#      Takes ONE image from the dataset and runs it through the augmentation
#      pipeline N times. Each run draws new random parameters, so you get N
#      different-looking versions of the same image. Seeing clear visual
#      differences (flips, rotations, brightness shifts, crops) is direct
#      proof that the transforms are firing.
#
#   2. show_original_vs_augmented()
#      Shows one image in its original state side-by-side with one augmented
#      version, PLUS the corresponding label mask for each. This confirms the
#      critical segmentation rule: the mask moves with the image. If you flip
#      the image, the mask is also flipped. The spatial alignment is preserved.
#
# WHEN TO CALL THESE:
#   Call them BEFORE training starts (Step 1.5 in main). This way you can
#   visually sanity-check the augmentation pipeline is wired up correctly
#   before committing to a long training run.
 
class AugmentationInspector:
    """
    Visual tools to confirm that data augmentation is applied correctly.
 
    WHY THIS MATTERS:
        If augmentation is misconfigured (e.g. mask not transformed, wrong
        probability values, transforms not actually firing), you won't see
        an error — training will just silently produce worse results.
        Visual inspection before training is the only reliable way to catch
        these issues early.
    """
 
    # Colour map for the 4 vegetation classes — used when drawing masks.
    # Each class gets a distinct colour so you can see class boundaries clearly.
    CLASS_COLOURS = {
        0: (70,  70,  70),    # Non-vegetal      → dark grey
        1: (144, 238, 144),   # Low vegetation   → light green
        2: (34,  139, 34),    # Moderate veg.    → forest green
        3: (0,   100, 0),     # Dense vegetation → dark green
    }
 
    def __init__(self, dataset: VegetationDataset, augmentation: Augmentation):
        """
        Args:
            dataset     : a VegetationDataset instance (used to load raw images)
            augmentation: an Augmentation instance to apply
        """
        self.dataset     = dataset
        self.augmentation = augmentation
 
    # -------------------------------------------------------------------------
    def _load_raw_pair(self, idx: int):
        """
        Loads the raw (un-augmented, un-normalised) image and NDVI class mask
        for dataset index `idx`.
 
        We bypass __getitem__ intentionally here — we want the PIL image and
        numpy mask BEFORE augmentation so we can show the true original.
 
        Returns:
            image_pil   : PIL Image   (H, W, 3)  — original RGB
            label_class : np.ndarray  (H, W)     — original class mask
        """
        sample_path, label_path = self.dataset.pairs[idx]
 
        # Load and resize image (no normalisation — we want display-ready values)
        image_np  = self.dataset._load_image(sample_path)
        image_pil = Image.fromarray(image_np).resize(
            (self.dataset.image_size[1], self.dataset.image_size[0]), Image.BILINEAR)
 
        # Load label → NDVI → class indices
        label_raw  = self.dataset._load_label_raw(label_path)
        label_ndvi = VegetationDataset.normalize_to_ndvi(
            label_raw, self.dataset.global_min, self.dataset.global_max)
 
        ndvi_u16   = ((label_ndvi + 1) / 2 * 65535).astype(np.uint16)
        ndvi_pil   = Image.fromarray(ndvi_u16).resize(
            (self.dataset.image_size[1], self.dataset.image_size[0]), Image.NEAREST)
        ndvi_r     = np.array(ndvi_pil).astype(np.float32) / 65535 * 2 - 1
 
        label_class = self.dataset._ndvi_to_class(ndvi_r)   # (H, W) int64
 
        return image_pil, label_class
 
    # -------------------------------------------------------------------------
    @staticmethod
    def _mask_to_rgb(mask: np.ndarray, class_colours: dict) -> np.ndarray:
        """
        Converts an integer class mask (H, W) into a colour image (H, W, 3).
 
        Each class index is mapped to its RGB colour so the mask is
        human-readable — instead of seeing pixel values 0/1/2/3 you see
        grey / light green / green / dark green.
        """
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls, colour in class_colours.items():
            rgb[mask == cls] = colour
        return rgb
 
    # -------------------------------------------------------------------------
    def show_augmentation_grid(self, idx: int = 0, n_versions: int = 8,
                                save_path: str = "augmentation_grid.png"):
        """
        Runs ONE image through the augmentation pipeline N times and displays
        all versions in a grid.
 
        Because augmentation is random, each run draws different parameter
        values. You should see clear variety: some images flipped, some rotated,
        some brighter or darker, some zoomed or shifted. If all images look
        identical, augmentation is not firing.
 
        Args:
            idx        : which dataset sample to use (default 0 = first image)
            n_versions : how many augmented versions to show
            save_path  : filename to save the grid image
        """
        image_orig, mask_orig = self._load_raw_pair(idx)
 
        # We want n_versions augmented images + 1 original = n_versions + 1 total
        # Arrange them in a grid with 2 rows: top = images, bottom = masks.
        n_cols = n_versions + 1   # first column = original, rest = augmented
 
        fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
        fig.suptitle(
            "Augmentation Grid — top: RGB image, bottom: class mask\n"
            "Column 0 = original. Columns 1+ = same image, different random augmentation.",
            fontsize=11
        )
 
        def _draw(col, img_pil, msk, title):
            """Helper: draw image and mask in the given column."""
            axes[0, col].imshow(np.array(img_pil))
            axes[0, col].set_title(title, fontsize=8)
            axes[0, col].axis('off')
 
            axes[1, col].imshow(self._mask_to_rgb(msk, self.CLASS_COLOURS))
            axes[1, col].axis('off')
 
        # Column 0: original (no augmentation)
        _draw(0, image_orig, mask_orig, "Original")
 
        # Columns 1…n: independently augmented versions
        for i in range(1, n_cols):
            # Each call to augmentation draws fresh random values — that is
            # the whole point. The same source image looks different every time.
            aug_img, aug_msk = self.augmentation(image_orig.copy(), mask_orig.copy())
            _draw(i, aug_img, aug_msk, f"Aug #{i}")
 
        # Add a colour legend for the mask row
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1,
                           color=tuple(c / 255 for c in col),
                           label=lbl)
            for lbl, col in zip(
                ["Non-vegetal", "Low veg.", "Moderate veg.", "Dense veg."],
                self.CLASS_COLOURS.values()
            )
        ]
        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=4, fontsize=8, title="Mask classes")
 
        plt.tight_layout(rect=[0, 0.06, 1, 1])  # leave room for legend
        plt.savefig(save_path, dpi=120)
        plt.show()
        print(f"[Inspector] Augmentation grid saved as '{save_path}'")
        print(f"[Inspector] If all {n_versions} augmented images look identical, "
              f"augmentation is NOT firing — check CONFIG['use_augmentation'] = True.")
 
    # -------------------------------------------------------------------------
    def show_original_vs_augmented(self, idx: int = 0,
                                    save_path: str = "augmentation_comparison.png"):
        """
        Shows one sample as a 2×2 grid:
            [original image]  |  [augmented image]
            [original mask ]  |  [augmented mask ]
 
        This specifically confirms the most critical segmentation requirement:
        the mask and image are transformed TOGETHER. You should see the mask
        rotate/flip/shift in exact lockstep with the image. Any misalignment
        here would mean labels are pointing at the wrong pixels — the worst
        possible error in a segmentation model.
 
        Args:
            idx       : dataset sample index
            save_path : filename to save
        """
        image_orig, mask_orig = self._load_raw_pair(idx)
        image_aug,  mask_aug  = self.augmentation(image_orig.copy(), mask_orig.copy())
 
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(
            "Original vs. Augmented — confirm image and mask move together\n"
            "Any spatial transform (flip, rotate, scale, translate) must appear "
            "identically in both columns.",
            fontsize=10
        )
 
        data = [
            (0, 0, np.array(image_orig), "Original image"),
            (0, 1, np.array(image_aug),  "Augmented image"),
            (1, 0, self._mask_to_rgb(mask_orig, self.CLASS_COLOURS), "Original mask"),
            (1, 1, self._mask_to_rgb(mask_aug,  self.CLASS_COLOURS), "Augmented mask"),
        ]
 
        for row, col, img_data, title in data:
            axes[row, col].imshow(img_data)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')
 
        # Colour legend
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1,
                           color=tuple(c / 255 for c in col),
                           label=lbl)
            for lbl, col in zip(
                ["Non-vegetal", "Low veg.", "Moderate veg.", "Dense veg."],
                self.CLASS_COLOURS.values()
            )
        ]
        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=4, fontsize=9, title="Mask classes")
 
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        plt.savefig(save_path, dpi=120)
        plt.show()
        print(f"[Inspector] Comparison saved as '{save_path}'")
        print("[Inspector] Check: do the image and mask share the exact same "
              "orientation, zoom, and position? They must.")
        
# =============================================================================
# CLASS 5: Trainer
# =============================================================================
# This is the EXPLICIT training loop — the PyTorch way.
# Unlike Keras's .fit(), we write every step ourselves.
# This lets you see exactly what happens in each iteration and modify anything.

class Trainer:
    """
    Manages the full training process over multiple epochs.

    Each epoch consists of:
        1. Training loop   → model sees training data, weights are updated
        2. Validation loop → model sees validation data, NO weight updates
                             (we only check performance to detect overfitting)

    Key concepts:
        Loss       : a number measuring how wrong the model is. We minimise this.
        Optimizer  : algorithm that adjusts weights to reduce the loss.
        Backprop   : computing HOW MUCH each weight contributed to the loss.
        Gradient   : the direction and magnitude to adjust each weight.
    """

    def __init__(self, model, train_loader, val_loader, config: dict):
        self.model       = model
        self.train_loader = train_loader
        self.val_loader  = val_loader
        self.config      = config
        self.device      = config["device"]

        # Move model to device (GPU if available, else CPU)
        self.model.to(self.device)

        # --- Loss function ---
        # CrossEntropyLoss is the standard for multi-class classification.
        # It measures how far the predicted class probabilities are from the true labels.
        # Internally: it applies LogSoftmax + NLLLoss.
        # Input:  (B, num_classes, H, W) raw logits + (B, H, W) int64 class indices
        # Output: a single scalar loss value
        self.loss_fn = nn.CrossEntropyLoss()
        # TODO: if your classes are imbalanced, pass weight=class_weights to CrossEntropyLoss
        #       class_weights = torch.tensor([w0, w1, w2]) where wi = 1/frequency_of_class_i

        # --- Optimizer ---
        # Adam is the most commonly used optimizer. It adapts the learning rate
        # per-parameter, making it robust and fast to converge.
        self.optimizer = optim.Adam(
            model.parameters(),                  # all learnable weights in the model
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]  # L2 penalty on large weights
        )

        # --- Learning rate scheduler (optional but helpful) ---
        # Reduces the learning rate when validation loss stops improving.
        # Think of it as: start with big steps, then take smaller steps as you
        # get closer to the optimal solution.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',         # 'min' because we want loss to go DOWN
            patience=5,         # wait 5 epochs without improvement before reducing LR
            factor=0.5          # multiply LR by 0.5 when triggered
        )

        # Storage for plotting training curves later
        self.train_losses = []
        self.val_losses   = []

    # -------------------------------------------------------------------------
    def train_one_epoch(self) -> float:
        """
        Runs ONE full pass through the training dataset.
        Returns the average loss across all batches.
        """
        self.model.train()  # IMPORTANT: puts model in "training mode"
                            # This enables Dropout (if any) and makes BatchNorm
                            # use batch statistics instead of running averages.

        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # Move data to the same device as the model (GPU/CPU)
            images = images.to(self.device)  # (B, 3, H, W)
            labels = labels.to(self.device)  # (B, H, W)  int64

            # ---- THE 5 STEPS OF A TRAINING ITERATION ----

            # Step 1: Forward pass — run images through the model
            predictions = self.model(images)    # (B, num_classes, H, W)

            # Step 2: Compute loss — how wrong are we?
            loss = self.loss_fn(predictions, labels)

            # Step 3: Zero gradients — CRUCIAL
            # PyTorch ACCUMULATES gradients by default (adds to whatever was already there).
            # If you forget this, gradients from the previous batch interfere with this batch.
            self.optimizer.zero_grad()

            # Step 4: Backward pass (backpropagation)
            # PyTorch automatically computes the gradient of the loss
            # with respect to every learnable parameter in the model.
            loss.backward()

            # Step 5: Update weights using the computed gradients
            self.optimizer.step()

            total_loss += loss.item()   # .item() extracts the scalar value from a 1-element tensor

        average_loss = total_loss / len(self.train_loader)
        return average_loss

    # -------------------------------------------------------------------------
    def validate_one_epoch(self) -> float:
        """
        Runs ONE pass through the validation dataset WITHOUT updating weights.
        Used to check if the model is generalising (not just memorising training data).
        """
        self.model.eval()   # Puts model in "evaluation mode"
                            # Disables Dropout, uses running averages in BatchNorm.

        total_loss = 0.0

        with torch.no_grad():
            # torch.no_grad() tells PyTorch NOT to track operations for gradient computation.
            # This saves memory and speeds up validation — we're not going to call .backward() here.
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(images)
                loss        = self.loss_fn(predictions, labels)
                total_loss += loss.item()

        average_loss = total_loss / len(self.val_loader)
        return average_loss

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Converts a raw second count into a readable string.

        Examples:
            45.3   → "45s"
            125.0  → "2m 05s"
            3725.0 → "1h 02m 05s"
        """
        seconds = int(seconds)
        h, remainder = divmod(seconds, 3600)
        m, s         = divmod(remainder, 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        elif m > 0:
            return f"{m}m {s:02d}s"
        else:
            return f"{s}s"

    # -------------------------------------------------------------------------
    def train(self) -> None:
        """
        Full training loop: runs train + validate for every epoch.
        Saves the best model (lowest validation loss) automatically.
        """
        print(f"\n[Trainer] Starting training on {self.device}")
        print(f"          {self.config['epochs']} epochs | "
              f"batch size {self.config['batch_size']} | "
              f"lr {self.config['learning_rate']}\n")

        best_val_loss    = float('inf')
        training_start   = time.time()         # wall-clock time when training begins

        for epoch in range(1, self.config["epochs"] + 1):

            epoch_start = time.time()          # time at the start of this epoch

            train_loss = self.train_one_epoch()
            val_loss   = self.validate_one_epoch()

            epoch_secs  = time.time() - epoch_start        # seconds this epoch took
            total_secs  = time.time() - training_start     # seconds since training began

            # Estimate how long is left by averaging seconds-per-epoch so far
            avg_secs_per_epoch = total_secs / epoch
            remaining_secs     = avg_secs_per_epoch * (self.config["epochs"] - epoch)

            # Tell the scheduler how validation is going — it may reduce LR
            self.scheduler.step(val_loss)

            # Store for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"  Epoch [{epoch:3d}/{self.config['epochs']}]  "
                  f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  "
                  f"Epoch time: {self._format_time(epoch_secs)}  |  "
                  f"Elapsed: {self._format_time(total_secs)}  |  "
                  f"ETA: {self._format_time(remaining_secs)}")

            # Save model if this is the best validation loss so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config["model_save_path"])
                print(f"             ✓ New best model saved (val loss: {val_loss:.4f})")

        total_training_time = time.time() - training_start
        print(f"\n[Trainer] Training complete in {self._format_time(total_training_time)}.")
        print(f"          Best val loss: {best_val_loss:.4f}")

    # -------------------------------------------------------------------------
    def plot_losses(self) -> None:
        """
        Plots training vs. validation loss curves over epochs.
        A healthy model shows both curves declining and staying close together.
        If training loss falls but val loss rises → OVERFITTING.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Training Loss",   color="steelblue")
        plt.plot(self.val_losses,   label="Validation Loss", color="darkorange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs. Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("loss_curve.png")
        plt.show()
        print("[Trainer] Loss curve saved as 'loss_curve.png'")


# =============================================================================
# CLASS 6: Evaluator
# =============================================================================
# After training, we evaluate the model on the TEST SET — data it has NEVER seen.
# This gives us an honest estimate of real-world performance.

class Evaluator:
    """
    Collects predictions on the test set and computes classification metrics.

    All metrics (accuracy, precision, recall, F1) come from comparing:
        y_true : the actual class labels (ground truth from the dataset)
        y_pred : the model's predicted class labels

    We flatten all pixel predictions across all test images into one big array
    before computing metrics — scikit-learn works on 1D arrays.
    """

    def __init__(self, model, test_loader, config: dict):
        self.model       = model
        self.test_loader = test_loader
        self.device      = config["device"]
        self.num_classes = config["num_classes"]

    # -------------------------------------------------------------------------
    def collect_predictions(self):
        """
        Runs the model on all test images and collects predictions.

        Returns:
            all_preds  : flat numpy array of predicted class indices
            all_labels : flat numpy array of true class indices
        """
        self.model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)

                # Forward pass: get raw scores for each class
                logits = self.model(images)    # (B, num_classes, H, W)

                # argmax along the class dimension → one class index per pixel
                preds = torch.argmax(logits, dim=1)  # (B, H, W)

                # Move to CPU and convert to numpy for scikit-learn
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())

        # Stack all batches and flatten all spatial dimensions into a 1D array
        # We go from [(B, H, W), (B, H, W), ...] → one big (N,) array
        all_preds  = np.concatenate([p.flatten() for p in all_preds])
        all_labels = np.concatenate([l.flatten() for l in all_labels])

        return all_preds, all_labels

    # -------------------------------------------------------------------------
    def evaluate(self, class_names: list = None) -> dict:
        """
        Computes and prints all evaluation metrics.

        METRICS EXPLAINED:
        ------------------
        Accuracy  : fraction of ALL pixels correctly classified
                    → simple but misleading if classes are imbalanced

        Precision : for each class, of all pixels WE called that class,
                    what fraction actually belonged to it?
                    → "when I say forest, am I usually right?"

        Recall    : for each class, of all pixels THAT ARE that class,
                    what fraction did we correctly label?
                    → "am I finding all the forest pixels?"

        F1-Score  : harmonic mean of Precision and Recall.
                    Punishes extreme imbalance between the two.
                    → the single best summary metric for classification

        Args:
            class_names : optional list of human-readable class names
                          e.g. ["Bare/Water", "Sparse Vegetation", "Dense Vegetation"]
        """
        print("\n[Evaluator] Running evaluation on test set...")
        y_pred, y_true = self.collect_predictions()

        # --- Overall Accuracy ---
        accuracy = accuracy_score(y_true, y_pred)

        # --- Per-class Precision, Recall, F1 ---
        # average='macro' → compute per class, then average equally (ignores class imbalance)
        # average='weighted' → same but weights by class frequency (good for imbalanced data)
        # zero_division=0 → if a class has no predictions, return 0 instead of warning
        precision = precision_score(y_true, y_pred, average='macro',    zero_division=0)
        recall    = recall_score(   y_true, y_pred, average='macro',    zero_division=0)
        f1        = f1_score(       y_true, y_pred, average='macro',    zero_division=0)

        # Full breakdown per class
        target_names = class_names if class_names else [f"Class {i}" for i in range(self.num_classes)]
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

        print("\n" + "=" * 60)
        print("  EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"  Precision : {precision:.4f}  (macro avg)")
        print(f"  Recall    : {recall:.4f}  (macro avg)")
        print(f"  F1-Score  : {f1:.4f}  (macro avg)")
        print("\n  Per-class breakdown:")
        print(report)
        print("=" * 60)

        # --- Confusion matrix ---
        self.plot_confusion_matrix(y_true, y_pred, target_names)

        return {
            "accuracy" : accuracy,
            "precision": precision,
            "recall"   : recall,
            "f1"       : f1,
        }

    # -------------------------------------------------------------------------
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: list) -> None:
        """
        Computes and displays the confusion matrix as a colour-coded heatmap.

        WHAT IS A CONFUSION MATRIX?
        ----------------------------
        It's a square grid of size (num_classes × num_classes).
        Each row represents one TRUE class; each column represents one
        PREDICTED class. The value in cell (i, j) is the number of pixels
        that ACTUALLY belong to class i but were PREDICTED as class j.

        Reading the matrix:
            • The main diagonal (top-left to bottom-right) shows correct
              predictions. You want these to be large.
            • Off-diagonal cells show mistakes. A large value at (i, j)
              means the model often confuses class i for class j — a useful
              clue for debugging (e.g., "it keeps mistaking moderate
              vegetation for dense vegetation").

        We plot TWO versions side by side:
            Left  — raw counts   : absolute number of pixels per cell
            Right — normalised   : each row divided by its total, so every
                                   row sums to 1.0. This is easier to read
                                   when classes have very different sizes,
                                   because a class with 10 000 pixels would
                                   visually dominate a raw-count plot.

        Args:
            y_true       : 1D array of true class indices
            y_pred       : 1D array of predicted class indices
            class_names  : list of human-readable class labels
        """
        # Compute the raw confusion matrix
        # Shape: (num_classes, num_classes) — integer counts
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))

        # Normalise each row so values are proportions (0.0 – 1.0)
        # A row that sums to 0 (class never appears) stays as 0 to avoid /0
        row_sums  = cm.sum(axis=1, keepdims=True)
        cm_normed = np.where(row_sums > 0, cm / row_sums, 0).astype(np.float32)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Confusion Matrix — Vegetation Classification", fontsize=14, fontweight='bold')

        for ax, data, title, fmt in zip(
            axes,
            [cm,       cm_normed],
            ["Raw counts", "Normalised (row %)"],
            ["d",      ".2f"]
        ):
            # imshow draws the matrix as a colour grid.
            # cmap='Blues': light blue = small values, dark blue = large values.
            im = ax.imshow(data, interpolation='nearest', cmap='Blues')
            plt.colorbar(im, ax=ax)

            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Predicted class", fontsize=10)
            ax.set_ylabel("True class",      fontsize=10)

            # Label both axes with the class names
            ticks = list(range(self.num_classes))
            ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=30, ha='right')
            ax.set_yticks(ticks); ax.set_yticklabels(class_names)

            # Print the numeric value inside each cell.
            # Text colour flips to white on dark cells so it stays readable.
            threshold = data.max() / 2.0
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    colour = "white" if data[i, j] > threshold else "black"
                    ax.text(j, i, format(data[i, j], fmt),
                            ha='center', va='center',
                            color=colour, fontsize=9)

        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        plt.show()
        print("[Evaluator] Confusion matrix saved as 'confusion_matrix.png'")


# =============================================================================
# MAIN FUNCTION
# =============================================================================
# Intentionally high-level: each line tells you WHAT is happening,
# not HOW. The HOW is inside the classes above.

def main():
    print("=" * 60)
    print("  VEGETATION CLASSIFICATION — DEEP LEARNING PIPELINE")
    print("=" * 60)

    # ── Step 0: Reproducibility ───────────────────────────────────────────────
    print("\n[0] Setting random seed for reproducibility...")
    set_seed(CONFIG["seed"])

    # ── Step 1: Build preprocessing and augmentation ──────────────────────────
    print("\n[1] Initialising preprocessing and augmentation...")
    preprocessing = Preprocessing()
    augmentation  = Augmentation() if CONFIG["use_augmentation"] else None

    # ── Step 2: Load dataset ──────────────────────────────────────────────────
    # The dataset pairs sample images with their label masks.
    # At this point, images are NOT loaded into memory — only the file paths are stored.
    # Images are loaded on-demand by the DataLoader during training (lazy loading).
    print("\n[2] Loading dataset from disk...")
    full_dataset = VegetationDataset(
        samples_dir      = CONFIG["samples_dir"],
        labels_dir       = CONFIG["labels_dir"],
        preprocessing    = preprocessing,
        augmentation     = augmentation,   # augmentation applied only to training data!
        image_size       = CONFIG["image_size"],
        ndvi_thresholds  = CONFIG["ndvi_thresholds"],  # [-1,1] cut-points: [0.0, 0.3, 0.6]
    )
    # TODO: Ideally, augmentation should only apply to training images, not validation/test.
    #       To handle this properly, create two dataset instances:
    #         train_dataset → with augmentation
    #         val_dataset   → without augmentation (same preprocessing, no augmentation)
    #       Then split by index rather than using random_split.

    # ── Step 2.5: Visually inspect augmentation ───────────────────────────────
    # This step is optional but strongly recommended before every training run.
    # It confirms that:
    #   (a) augmentation transforms are actually firing
    #   (b) the image and mask stay spatially aligned after every transform
    #
    # The parameter count is ALWAYS the same with or without augmentation —
    # augmentation changes the data, not the model. The only way to confirm
    # it is working is to look at the images directly.
    #
    # Two plots are saved:
    #   augmentation_grid.png       → one image, 8 random augmented versions
    #   augmentation_comparison.png → original vs augmented, image + mask side by side
    if augmentation is not None:
        print("\n[2.5] Inspecting augmentation output...")
        inspector = AugmentationInspector(full_dataset, augmentation)
        inspector.show_augmentation_grid(idx=0, n_versions=8)
        inspector.show_original_vs_augmented(idx=0)
    else:
        print("\n[2.5] Augmentation is OFF — skipping visual inspection.")

    # ── Step 3: Split into train / val / test ────────────────────────────────
    print("\n[3] Splitting data into train / val / test sets...")
    total       = len(full_dataset)
    n_train     = int(total * CONFIG["train_ratio"])
    n_val       = int(total * CONFIG["val_ratio"])
    n_test      = total - n_train - n_val       # remainder goes to test

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(CONFIG["seed"])
        # fixing the generator seed ensures the same split every run
    )

    print(f"    Total: {total} | Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # ── Step 4: Create DataLoaders ────────────────────────────────────────────
    # DataLoader wraps a Dataset and handles:
    #   - Batching: groups samples into batches of batch_size
    #   - Shuffling: randomises order each epoch (only for training)
    #   - Parallel loading: num_workers processes load data in the background
    print("\n[4] Creating DataLoaders...")
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)
    # pin_memory=True: keeps data in pinned (page-locked) CPU memory for faster GPU transfer.
    # Only useful if device is CUDA. Harmless on CPU.

    # ── Step 5: Build the model ───────────────────────────────────────────────
    # print("\n[5] Building the model...")
    # TODO: Create the model instance
    # print("\n[5] Building the model...")
    # model = VegetationModel(
    #     in_channels  = 3,                       # RGB
    #     num_classes  = CONFIG["num_classes"],
    #     features     = [32, 64, 128, 256]        # TODO: try [64, 128, 256, 512] for larger model
    # )

    # Count and display the number of learnable parameters
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"    Model has {n_params:,} trainable parameters.")

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    # print("\n[6] Training the model...")
    # trainer = Trainer(model, train_loader, val_loader, CONFIG)
    # trainer.train()
    # trainer.plot_losses()

    # ── Step 7: Load the best saved model weights ─────────────────────────────
    # We always evaluate using the BEST checkpoint (lowest val loss),
    # not the weights from the final epoch (which may have overfit slightly).
    # print("\n[7] Loading best model weights for evaluation...")
    # model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=CONFIG["device"]))

    # ── Step 8: Evaluate on the test set ──────────────────────────────────────
    # print("\n[8] Evaluating on the held-out test set...")
    # evaluator = Evaluator(model, test_loader, CONFIG)
    # metrics   = evaluator.evaluate(
    #     class_names=["Non-vegetal", "Low Vegetation", "Moderate Vegetation", "Dense Vegetation"]
    # )

    print("\n[Pipeline complete]")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # This block only runs when you execute this file directly:
    #     python vegetation_classification.py
    # It does NOT run if this file is imported as a module from another script.
    main()