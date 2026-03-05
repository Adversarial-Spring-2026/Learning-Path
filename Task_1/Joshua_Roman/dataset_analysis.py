import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

IMAGE_DIR = "Learning_Path/samples"
LABEL_DIR = "Learning_Path/labels"

image_files = sorted(os.listdir(IMAGE_DIR))

print(f"Total images found: {len(image_files)}")

def data_quality_control():
    print("\nRunning data quality control...")
    print("--------------------------------")

    issues_found = 0

    for image_filename in image_files:
        #Finds the designated label file for the current image file
        label_filename = image_filename.replace("img", "ndvi")
        label_path = os.path.join(LABEL_DIR, label_filename)

        #Label file not found
        if not os.path.exists(label_path):
            print(f"Missing label file for image: {image_filename}")
            issues_found += 1
            continue

        #Verify both image and label reolution
        img_path = os.path.join(IMAGE_DIR, image_filename)
        with rasterio.open(img_path) as src:
            img_meta = src.meta
            img_width, img_height = img_meta['width'], img_meta['height']
        with rasterio.open(label_path) as src:
            label_meta = src.meta
            label_width, label_height = label_meta['width'], label_meta['height']
        if (img_width, img_height) != (label_width, label_height):
            print(f"Resolution mismatch for {image_filename}:")
            print(f"  Image resolution: {img_width}x{img_height}")
            print(f"  Label resolution: {label_width}x{label_height}")
            issues_found += 1

    #End of quality control
    if not issues_found:
        print("\nData quality control complete! No issues found.")
    else:
        print("\nData quality control complete! Number of issues found: ", issues_found)
    
#Loads sample with corresponding label
def load_pair(image_filename):
    img_path = os.path.join(IMAGE_DIR, image_filename)
    label_filename = image_filename.replace("img", "ndvi")
    label_path = (os.path.join(LABEL_DIR, label_filename))

    with rasterio.open(img_path) as src:
        rgb = src.read().astype("float32")
    
    rgb = np.transpose(rgb, (1, 2, 0))
    #Normalize RGB for accurate full color image
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))

    with rasterio.open(label_path) as src:
        ndvi = src.read(1).astype("float32")

    return rgb, ndvi

#Visualize sample-label pairs together and mean NDVI indicator
def sample_viewer():
    index = 0
    total = len(image_files)

    print(f"\nDataset contains {total} samples.")
    print("Commands:")
    print(" index #  -> jump to index")
    print("    n     -> next image")
    print("    p     -> previous image")
    print("    q     -> quit\n")

    #Starting at index 0
    while True:
        file = image_files[index]
        rgb, ndvi = load_pair(file)
        #---NDVI STATS FOR THIS SAMPLE---
        print(f"\nViewing Index: {index}")
        print(f"Min NDVI: {np.min(ndvi):.3f}")
        print(f"Max NDVI: {np.max(ndvi):.3f}")
        print(f"Mean NDVI: {np.mean(ndvi):.3f}")

        plt.close("all")

        plt.figure(figsize=(12, 5))

        #Sample image
        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.title(f"True Color Image (Index {index})")
        plt.axis("off")

        #NDVI image
        plt.subplot(1, 2, 2)
        im = plt.imshow(ndvi, cmap="viridis")
        plt.title("NDVI Label")

        #MEAN NDVI indicator
        cbar = plt.colorbar(im)
        cbar.ax.hlines(np.mean(ndvi), xmin=0, xmax=1, color="red", linewidth=2)

        plt.show()

        #User decision after viewing last image
        user_input = input("\nEnter command: ")
        if user_input.lower() == 'q':
            print("Exiting viewer...")
            break
        elif user_input.lower() == 'n':
            index = (index + 1) % total

        elif user_input.lower() == 'p':
            index = (index - 1) % total

        elif user_input.isdigit():
            new_index = int(user_input)
            if new_index < 0 or new_index >= total:
                print("Index out of range.")
            else:
                index = new_index

        else:
            print("Invalid input. Please enter a number.")

#Print overall DATASET statistics    
def dataset_ndvi_distribution():
    all_pixels = []

    for file in image_files:
        _, ndvi = load_pair(file)
        all_pixels.append(ndvi.flatten())

    all_pixels = np.concatenate(all_pixels)

    print("\nNDVI Statistics Across Dataset")
    print("--------------------------------")
    print("Min NDVI:", np.min(all_pixels))
    print("Max NDVI:", np.max(all_pixels))
    print("Mean NDVI:", np.mean(all_pixels))
    print("Std Dev NDVI:", np.std(all_pixels))

    return all_pixels

#MAIN
if __name__ == "__main__":
    data_quality_control()

    sample_viewer()

    all_pixels = dataset_ndvi_distribution()

    print("\nDataset analysis complete!")

        


    