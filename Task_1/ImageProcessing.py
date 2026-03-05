from pathlib import Path
import rasterio #import rasterio to read the Tiff images and data
import numpy as np #import numpy to handle the math
import matplotlib.pyplot as plt #imported matplot to help with the display of images
TASK1_DIR = Path(__file__).resolve().parent # folder: Learning_Path / Task 1
BASE_DIR = TASK1_DIR.parent # folder: Learning_Path

samples_dir = BASE_DIR / "samples" #Essentially becomes Learning_Path/samples
labels_dir = BASE_DIR / "labels" #Same here with Learning_Path/labels

##Directories, with printed paths for better visualization
print("TASK1_DIR:", TASK1_DIR) 
print("BASE_DIR :", BASE_DIR)
print("samples  :", samples_dir)
print("labels   :", labels_dir)
##Check if the folders exists using .exists()
print("samples exists?", samples_dir.exists())
print("labels  exists?", labels_dir.exists())
#Check for files that start with anything and end with .tif
sample_files = sorted(samples_dir.glob("*.tif*"))
label_files = sorted(labels_dir.glob("*.tif*"))
#Get total Samples and Labels of how many tif files there are in the folders
print("\nTotal Samples: ", len(sample_files))
print("Total labels: ", len(label_files))
print("\nFirst 3 Samples")
##Print first few (3) samples and labels
for s in sample_files[:3]:
    print(" -", s.name)
print("\nFirst 3 Labels")
for l in label_files[:3]:
    print(" -", l.name)

#Choose index

index = 0 #always chooses the first index, but can be changed to another if necesarry
image_path = sample_files[index]
#Assist Error Handling / Initialization of new Variables
label_name = image_path.name.replace("img_", "ndvi_")
label_path = labels_dir / label_name
#Error handling
if not label_path.exists():
    raise FileNotFoundError(f"Label not found for {image_path.name} -> {label_path.name}")

#Read image data using the rasterio library

with rasterio.open(image_path) as src:
    image = src.read() # (bands, Height, Width)

with rasterio.open(label_path) as src:
    label = src.read(1) # (Height, Width)
#Print information about the data
print("Sample shape:", image.shape, image.dtype)
print("Label  shape:", label.shape, label.dtype)
print("Label min/max:", float(np.nanmin(label)), float(np.nanmax(label)))

#Build rgb image

rgb = image[:3].astype(np.float32)
rgb = np.transpose(rgb, (1,2,0)) #use numpy to complete matrix operations

#Display
rgb_disp = rgb.copy() # normalize the RGB for better display
for c in range(3):
    low, hi = np.percentile(rgb_disp[:,:,c], (2,98))
    if hi > low:
        rgb_disp[:,:,c] = np.clip((rgb_disp[:,:,c]-low)/(hi-low), 0, 1)
    else:
        rgb_disp[:,:,c] = 0

#Plot
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(rgb_disp)
plt.title("RGB")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(image[0], cmap="gray")  #can use gray or viridris, should use viridis as it is more modern
plt.title("Band 1")
plt.axis("off")

plt.subplot(1,3,3)
im = plt.imshow(label, cmap="gray") #sme here
plt.title("NDVI Label")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.axis("off")

plt.tight_layout()
plt.show()