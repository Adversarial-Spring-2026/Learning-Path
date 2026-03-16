import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import os


SAMPLES_DIR = "../samples"
LABELS_DIR  = "../labels"


def load_and_show(image_path, label_path):
    with rasterio.open(image_path) as src:
        image = src.read()          
        meta  = src.meta
        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"  Bands: {src.count}, Size: {src.width}x{src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  Dtype: {src.dtypes[0]}")

   
    with rasterio.open(label_path) as lbl:
        label = lbl.read(1)        
        print(f"  Label shape: {label.shape}")
        print(f"  Label unique values: {np.unique(label)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if image.shape[0] >= 3:
        rgb = np.stack([image[2], image[1], image[0]], axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) 
        axes[0].imshow(rgb)
    else:
        axes[0].imshow(image[0], cmap="gray")
    axes[0].set_title("Sample Image (RGB)")
    axes[0].axis("off")

    axes[1].imshow(label, cmap="RdYlGn")
    axes[1].set_title("NDVI Label")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

sample_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(".tif") or f.endswith(".tiff")])
label_files  = sorted([f for f in os.listdir(LABELS_DIR)  if f.endswith(".tif") or f.endswith(".tiff")])

print(f"Total images: {len(sample_files)}")
print(f"Total labels: {len(label_files)}")

if len(sample_files) != len(label_files):
    print("Mismatch")

for i in range(min(3, len(sample_files))):
    load_and_show(
        os.path.join(SAMPLES_DIR, sample_files[i]),
        os.path.join(LABELS_DIR,  label_files[i])
    )