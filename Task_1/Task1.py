import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROUTE_SAMPLES = "data/samples"
ROUTE_LABELS  = "data/labels"

files  = [f for f in os.listdir(ROUTE_SAMPLES) if f.endswith(".tiff")]
paired = [f for f in files if os.path.exists(os.path.join(ROUTE_LABELS, f.replace("_img_", "_ndvi_")))]

N = 5 #las N imagenes que quieres que aparescan al correr el archivo N > 2; 
fig, axes = plt.subplots(N, 3, figsize=(15, N * 4))

for i, img_name in enumerate(paired[:N]):
    img_path = os.path.join(ROUTE_SAMPLES, img_name)
    lbl_path = os.path.join(ROUTE_LABELS, img_name.replace("_img_", "_ndvi_"))

    with rasterio.open(img_path) as si, rasterio.open(lbl_path) as sl:
        r = si.read(1).astype(float)
        g = si.read(2).astype(float)
        b = si.read(3).astype(float)
        max_val = max(r.max(), g.max(), b.max())
        rgb = np.stack([r, g, b], axis=-1) / max_val

        ndvi = sl.read(1).astype(float)
        if ndvi.max() > 1 or ndvi.min() < -1:
            ndvi = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min()) * 2 - 1

    plant_pct = (ndvi > 0.3).mean() * 100

    im0 = axes[i, 0].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[i, 0].set_title(f"Vegetation Health — {plant_pct:.1f}% plants")
    axes[i, 0].axis("off")
    plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

    axes[i, 1].imshow(rgb)
    axes[i, 1].set_title("Original Image")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(ndvi / ndvi.max(), cmap="gray")
    axes[i, 2].set_title("NDVI (raw)")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
