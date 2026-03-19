import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image_path = "samples/"
labels_path = "labels/"

for image_file in os.listdir(image_path):
    if "_img_" in image_file:  
        label_file= image_file.replace("_img_", "_ndvi_") 
        image_full_path= os.path.join(image_path, image_file)
        label_full_path= os.path.join(labels_path, label_file)

        image= Image.open(image_full_path)
        label= Image.open(label_full_path)

        plt.figure(figsize=(8,4)) 
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title("img")
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(label, cmap='gray')
        plt.title("lbl")
        plt.axis("off")   
        plt.show()
        
        image_array = np.array(image)
        label_array = np.array(label)

        print("_____________________________________________________")
        print("\nShowing pair of label and image :")
        print("Image shape:",image_array.shape)
        print("Label shape:",label_array.shape)
        print("Image min/max:",np.min(image_array), np.max(image_array))
        print("Label min/max:",np.min(label_array), np.max(label_array))

        input("Press enter in terminal to see next pair")
    