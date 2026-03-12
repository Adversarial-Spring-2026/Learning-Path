import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

#Insert your own path to the datasets here
SAMPLES_DIR = ".../samples" 
LABELS_DIR = ".../labels"

files = sorted(os.listdir(SAMPLES_DIR))
print(f"Total images found: {len(files)}")


def load_pair(fname):
    img_path = os.path.join(SAMPLES_DIR, fname)
    lbl_path = os.path.join(LABELS_DIR, fname.replace("img", "ndvi"))

    with rasterio.open(img_path) as src:
        img = src.read().astype("float32")
        meta = src.meta

    img = np.transpose(img, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    with rasterio.open(lbl_path) as src:
        ndvi = src.read(1).astype("float32")

    return img, ndvi, meta


def data_cleaning():
    bugs = 0

    for fname in files:
        lbl_path = os.path.join(LABELS_DIR, fname.replace("img", "ndvi"))

        if not os.path.exists(lbl_path):
            print(f"  Missing label: {fname}")
            bugs += 1
            continue

        with rasterio.open(os.path.join(SAMPLES_DIR, fname)) as src:
            img_w, img_h = src.width, src.height
        with rasterio.open(lbl_path) as src:
            lbl_w, lbl_h = src.width, src.height

        if (img_w, img_h) != (lbl_w, lbl_h):
            print(f"  Size mismatch {fname}: image {img_w}x{img_h} vs label {lbl_w}x{lbl_h}")
            bugs += 1

    if bugs == 0:
        print("No bugs found.")
    else:
        print(f"{bugs} bugs found.")


def inspect_metadata():
    print("\nMetadata for first image")
    with rasterio.open(os.path.join(SAMPLES_DIR, files[0])) as src:
        meta = src.meta
        print(f"  Driver:    {meta['driver']}")
        print(f"  Size:      {meta['width']} x {meta['height']}")
        print(f"  Bands:     {meta['count']}")
        print(f"  Dtype:     {meta['dtype']}")
        print(f"  CRS:       {src.crs}")
        print(f"  Transform: {src.transform}")


def dataset_statistics():
    print("\nComputing NDVI metric...")
    pixels = np.concatenate([load_pair(fname)[1].ravel() for fname in files])

    print(f"  Min:  {pixels.min():.4f}")
    print(f"  Max:  {pixels.max():.4f}")
    print(f"  Mean: {pixels.mean():.4f}")
    print(f"  Std:  {pixels.std():.4f}")

    plt.figure(figsize=(8, 4))
    plt.hist(pixels, bins=100, color="seagreen", edgecolor="none")
    plt.axvline(pixels.mean(), color="red", linestyle="--",
                label=f"mean = {pixels.mean():.3f}")
    plt.title("NDVI Distribution across Dataset")
    plt.xlabel("NDVI")
    plt.ylabel("Pixel count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def simple_manipulations():
    print("\nSimple manipulations on first sample")

    img, ndvi, _ = load_pair(files[0])

    img_crop  = img[:128, :128]
    ndvi_crop = ndvi[:128, :128]
    print(f"  Original size: {img.shape[1]}x{img.shape[0]}")
    print(f"  Cropped size:  {img_crop.shape[1]}x{img_crop.shape[0]}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_crop)
    axes[0].set_title("Cropped RGB")
    axes[0].axis("off")
    axes[1].imshow(ndvi_crop, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("Cropped NDVI")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def sample_viewer():
    total = len(files)
    index = 0

    print(f"\nDataset contains {total} samples.")
    print("Commands:  n=next  p=previous  index <#>=jump  q=quit\n")

    while True:
        img, ndvi = load_pair(files[index])[:2]

        print(f"Index {index} | {files[index]}")
        print(f"  min={ndvi.min():.3f}  max={ndvi.max():.3f}  mean={ndvi.mean():.3f}")

        plt.close("all")
        fig, (ax_img, ax_ndvi) = plt.subplots(1, 2, figsize=(12, 5))

        ax_img.imshow(img)
        ax_img.set_title(f"True Color (index {index})")
        ax_img.axis("off")

        shown = ax_ndvi.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
        ax_ndvi.set_title("NDVI Label")
        ax_ndvi.axis("off")

        cbar = plt.colorbar(shown, ax=ax_ndvi, fraction=0.046, pad=0.04)
        cbar.ax.axhline(ndvi.mean(), color="blue", linewidth=2)

        plt.tight_layout()
        plt.show()

        cmd = input("Command: ").strip().lower()
        if cmd == "q":
            print("Exiting viewer.")
            break
        elif cmd == "n":
            index = (index + 1) % total
        elif cmd == "p":
            index = (index - 1) % total
        elif cmd.startswith("index"):
            try:
                new_index = int(cmd.split()[1])
                if 0 <= new_index < total:
                    index = new_index
                else:
                    print("Index out of range.")
            except (IndexError, ValueError):
                print("Usage: index <number>")
        else:
            print("Unknown command.")


if __name__ == "__main__":
    data_cleaning()
    inspect_metadata()
    dataset_statistics()
    simple_manipulations()
    sample_viewer()
