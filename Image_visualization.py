import os
import matplotlib.pyplot as plt
from PIL import Image

folder1 = "labels"
folder2 = "samples"

images1 = os.listdir(folder1)
images2 = os.listdir(folder2)

min_length = min(len(images1), len(images2))
# images1 = images1[:min_length]
# images2 = images2[:min_length]

index = 0

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.2)

def compare_images():
    if len(images1) != len(images2):
        print("Missing labels") if min_length == len(images1) else print("Missing samples")

    for i in range(len(images1)):
        print(images1[i].replace("ndvi", "") == images2[i].replace("img", ""))

def show_images():
    axes[0].clear()
    axes[1].clear()

    img1_path = os.path.join(folder1, images1[index])
    img2_path = os.path.join(folder2, images2[index])

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    axes[0].imshow(img1)
    axes[1].imshow(img2)

    axes[0].set_title(images1[index], fontsize=5)
    axes[1].set_title(images2[index], fontsize=5)

    axes[0].axis("off")
    axes[1].axis("off")

    fig.canvas.draw()

def on_key(event):
    global index

    if event.key == "right":
        index = (index + 1) % min_length
        show_images()

    elif event.key == "left":
        index = (index - 1) % min_length
        show_images()
    

fig.canvas.mpl_connect("key_press_event", on_key)

compare_images()
show_images()
plt.show()