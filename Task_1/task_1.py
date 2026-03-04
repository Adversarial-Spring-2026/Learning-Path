import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# dataset folders
samples_dir = r"C:\Users\Public\OpenFrameworks\apps\myApps\PandaHat\Learning_Path\samples"
labels_dir  = r"C:\Users\Public\OpenFrameworks\apps\myApps\PandaHat\Learning_Path\labels"

# list images
images = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith((".tif", ".tiff"))])

if not images:
    raise FileNotFoundError("No .tif/.tiff files found in samples folder.")

print("Available images:")
for i, img in enumerate(images):
    print(i, "-", img)

# choose image
idx = int(input("\nSelect image index: "))
if idx < 0 or idx >= len(images):
    raise ValueError("Invalid image index.")

image_name = images[idx]

# build paths
image_path = os.path.join(samples_dir, image_name)
label_name = image_name.replace("img_", "ndvi_")  
label_path = os.path.join(labels_dir, label_name)

print("\nLoading image:", image_name)
print("Loading label:", label_name)

# load image
with rasterio.open(image_path) as src:
    image = src.read()  

# load label
with rasterio.open(label_path) as src:
    label = src.read(1)  

# choose band 
print("\nThis image has", image.shape[0], "bands (0 to", image.shape[0] - 1, ")")
band = int(input("Choose band index to display: "))

if band < 0 or band >= image.shape[0]:
    print("Invalid band. Using band 0.")
    band = 0

# create RGB composite 
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
if image.shape[0] >= 3:
    rgb = image[:3]
    rgb = np.transpose(rgb, (1, 2, 0))
    max_val = np.max(rgb)
    if max_val != 0:
        rgb = rgb / max_val
    plt.imshow(rgb)
    plt.title("RGB Image (bands 1-3)")
else:
    plt.imshow(image[0], cmap="gray")
    plt.title("Not enough bands for RGB")
plt.axis("off")

# chosen band
plt.subplot(1, 3, 2)
plt.imshow(image[band], cmap="gray")
plt.title(f"Band {band + 1}")
plt.axis("off")

# NDVI label
plt.subplot(1, 3, 3)
plt.imshow(label, cmap="viridis")
plt.title("NDVI Label")
plt.axis("off")
plt.colorbar()

# print stats
print("\nImage shape (bands,H,W):", image.shape)
print("Label shape (H,W):", label.shape)
print("Image min/max:", np.min(image), np.max(image))
print("Label (NDVI) min/max:", np.min(label), np.max(label))

plt.show()