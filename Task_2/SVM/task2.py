import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd

# Rescale the data, we go from 86-219 to -1-1, so it match the ndvi real values
# Then we divide it into 4 classes where:
# -1-0: no vegetation
# 0-0.2: dry soil, little vegetation
# 0.2-0.5: grass, crops, medium vegetation
# 0.5-1: A lot of vegetation, healthy
#
# Downside, something to keep in mind
# We dont see different water, street or buildings, all of those are lable as 0

class SentinelDataset:
    def __init__(self, samples_folder, labels_folder):
        self.samples_folder = samples_folder
        self.labels_folder = labels_folder
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.scaler = None

    # --- Helper: normalize all images ---
    @staticmethod
    def compute_global_ndvi_range(labels_folder):
        min_val = float('inf')
        max_val = float('-inf')

        label_files = sorted(os.listdir(labels_folder))
        for l_file in label_files:
            with rasterio.open(os.path.join(labels_folder, l_file)) as src:
                label = src.read(1).astype(np.float32)
                min_val = min(min_val, label.min())
                max_val = max(max_val, label.max())

        print(f"Global NDVI range: min={min_val}, max={max_val}")
        return min_val, max_val
    
    # --- Helper: normalize label to NDVI ---
    @staticmethod
    def normalize_to_ndvi(label, global_min, global_max):
        """
        Rescale the label value into a range of (-1, 1)
        """
        # Convert to float to avoid integer issues
        label = label.astype(np.float32)

        # Rescale to [-1, 1]
        ndvi = (label - global_min) / (global_max - global_min) * 2 - 1
        return ndvi

    # --- Helper: convert NDVI to classes ---
    @staticmethod
    def ndvi_to_classes(ndvi):
        """
        Divide the ndvi values into clasess where:
        [-1,0]: non-vegetation
        [0,0.2]: low vegetation
        [0.2, 0.5]: moderate vegetation
        [0.5, 1]: dense vegetation
        """
        classes = np.zeros_like(ndvi, dtype=np.uint8)

        # Define thresholds
        classes[ndvi < 0] = 0                      # non-vegetation
        classes[(ndvi >= 0) & (ndvi < 0.2)] = 1    # low vegetation
        classes[(ndvi >= 0.2) & (ndvi < 0.5)] = 2  # moderate vegetation
        classes[ndvi >= 0.5] = 3                   # dense vegetation

        return classes

    # --- Load one image pair ---
    @staticmethod
    def load_image_pair(sample_path, label_path):
        with rasterio.open(sample_path) as src:
            sample = src.read()  # (bands, height, width)
        with rasterio.open(label_path) as src:
            label = src.read(1)  # (height, width)
        return sample, label

    # --- Build features and labels for one image ---
    @staticmethod
    def process_image(sample, label, global_min, global_max):
        # Target (y): Derived ONLY from the label file
        label_ndvi = SentinelDataset.normalize_to_ndvi(label, global_min, global_max)
        classes = SentinelDataset.ndvi_to_classes(label_ndvi)

        # Features (X): Derived from the raw sample image
        bands, h, w = sample.shape
        sample_flat = sample.reshape(bands, -1).T   # (pixels, bands)
        
        # Add NDVI from the color, might not be so accurate as with NIR
        R = sample_flat[:, 0].astype(np.float32)
        G = sample_flat[:, 1].astype(np.float32)
        B = sample_flat[:, 2].astype(np.float32)

        vari = (G - R) / (G + R - B + 1e-6)  # (pixels, 1)
        vari_flat = vari.reshape(-1, 1)

        X = np.hstack((sample_flat, vari_flat))  # (pixels, 4)
        # X = np.hstack((sample_flat))
        # X = sample_flat
        y = classes.reshape(-1)

        return X, y

    # --- Build dataset from folders ---
    def build_dataset(self, max_train_samples=100000, max_test_samples=50000, test_size=0.2, random_state=42, max_images=None):
        sample_files = sorted(os.listdir(self.samples_folder))
        label_files = sorted(os.listdir(self.labels_folder))

        if max_images is not None:
            sample_files = sample_files[:max_images]
            label_files = label_files[:max_images]

        # Calculate global ndvi range
        global_min, global_max = SentinelDataset.compute_global_ndvi_range(self.labels_folder)

        # Split data at file level (Prevents spatial autocorrelation)
        file_pairs = list(zip(sample_files, label_files))
        train_pairs, test_pairs = train_test_split(file_pairs, test_size=test_size, random_state=random_state)

        print(f"Processing {len(train_pairs)} training images...")
        X_train_list = []
        y_train_list = [] 

        for s_file, l_file in train_pairs:
            s_path = os.path.join(self.samples_folder, s_file)
            l_path = os.path.join(self.labels_folder, l_file)
            
            sample, label = self.load_image_pair(s_path, l_path)
            X_img, y_img = self.process_image(sample, label, global_min, global_max)
            
            X_train_list.append(X_img)
            y_train_list.append(y_img)

        self.X_train = np.vstack(X_train_list)
        self.y_train = np.hstack(y_train_list)

        X_test_list = []
        y_test_list = [] 

        print(f"Processing {len(test_pairs)} test images...")
        for s_file_t, l_file_t in test_pairs:
            s_path = os.path.join(self.samples_folder, s_file_t)
            l_path = os.path.join(self.labels_folder, l_file_t)
            
            sample, label = self.load_image_pair(s_path, l_path)
            X_img, y_img = self.process_image(sample, label, global_min, global_max)
            
            X_test_list.append(X_img)
            y_test_list.append(y_img)

        self.X_test = np.vstack(X_test_list)
        self.y_test = np.hstack(y_test_list)


        # SUBSAMPLING (Prevents SVM from crashing/taking hours)
        n_samples = self.X_train.shape[0]
        if n_samples > max_train_samples:
            print(f"Dataset too large ({n_samples} pixels), subsampling to {max_train_samples} pixels...")
            idx = np.random.choice(n_samples, size=max_train_samples, replace=False)
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]
            print("Subsampling done.")

        n_test_samples = self.X_test.shape[0]

        if n_test_samples > max_test_samples:
            print(f"Test set too large ({n_test_samples} pixels), subsampling to {max_test_samples}...")
            idx = np.random.choice(n_test_samples, size=max_test_samples, replace=False)
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx]
        print("Test subsampling done.")

        return self.X_train, self.y_train, self.X_test, self.y_test

    # --- Scale features (important for SVM) ---
    def scale_features(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        return self.X_train

class Model:
    def __init__(self):
        # kernel='rbf' → handles nonlinear boundaries
        # C=1.0 → default regularization
        # gamma='scale' → scales kernel automatically based on features
        # class_weight='balanced' → automatically compensates if some classes dominate
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', verbose=True)

        # 0,5  75
        # 0,8 76
        # 1.0 
    def train_and_evaluate_svm(self, X, y, X_test, y_test):
        """
        Train an SVM on X, y and evaluate its performance.

        Args:
            test_size: % of test size 
        """
        # n_samples = X.shape[0]
        # if n_samples > max_samples:
        #     print(f"Dataset too large ({n_samples} pixels), subsampling to {max_samples} pixels...")
        #     idx = np.random.choice(n_samples, size=max_samples, replace=False)
        #     X = X[idx]
        #     y = y[idx]
        #     print("Subsampling done.")
        # else:
        #     print(f"Dataset size is {n_samples}, using full dataset.")

        # --- Train ---
        print("Training SVM...")
        self.model.fit(X, y)
        print("Training complete.")

        # --- Predict on test set ---
        print("Making predictions...")
        y_pred = self.model.predict(X_test)
        print("Done making predictions!")

        # --- Evaluate ---
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        print("Classification Report Table:")
        print(report_df)

        # --- Confusion matrix as table ---
        labels = [0, 1, 2, 3]  # your class labels
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, 
                            index=[f"True {i}" for i in labels],
                            columns=[f"Pred {i}" for i in labels])
        
        print("\nConfusion Matrix Table:")
        print(cm_df)

        # --- Colored heatmap ---
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        plt.show()

        return X_test, y_test, y_pred

if __name__ == "__main__":
    print("Creating object...")
    dataset = SentinelDataset("samples", "labels")
    print("Create object complete")
    print("Creating dataset...")
    dataset.build_dataset()
    dataset.scale_features()
    print("Dataset ready.")

    print("Training model...")
    model = Model()
    model.train_and_evaluate_svm(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)

    # To save the model and reuse it
    # Save the model
    joblib.dump(model.model, "svm_model.pkl") # Note: dumping model.model to just save the SVC
    # Save the scaler too (needed for new data)
    joblib.dump(dataset.scaler, "scaler.pkl")

    # Load SVM and scaler
    # model = joblib.load("svm_model.pkl")
    # scaler = joblib.load("scaler.pkl")

    # # Example: predict on new features
    # X_new_scaled = scaler.transform(X_new)
    # y_new_pred = model.predict(X_new_scaled)
