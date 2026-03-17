from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime

# Global
# change this variable to test more/less samples
SAMPLE_TEST_AMOUNT = 5 

print("="*80)
print("STARTING DATASET ANALYSIS")
print("="*80)
print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Sample test amount: {SAMPLE_TEST_AMOUNT}") 
print()


# Directories
base_dir = "/Users/revelvelazquez/Pandahat Adverserial/Learning-Path/"
samples_path = os.path.join(base_dir, "samples/")
labels_path = os.path.join(base_dir, "labels/")
output_dir = os.path.join(base_dir, "analysis_output")
os.makedirs(output_dir, exist_ok=True)

# Stores all data found
report = []


# Loading the data and verification
print("\nLoading and Verifying Dataset")
print("="*50)

sample_files = sorted(glob.glob(os.path.join(samples_path, "*.tiff")))
label_files = sorted(glob.glob(os.path.join(labels_path, "*.tiff")))

dataset_summary = f"""
DATASET OVERVIEW
==========================
Samples folder: {samples_path}
Labels folder: {labels_path}
Total sample files: {len(sample_files)}
Total label files: {len(label_files)}
==========================
"""
if len(sample_files) == len(label_files):
    dataset_summary += "Sample and label counts MATCH\n"
else:
    dataset_summary += f"WARNING: Count mismatch! Samples: {len(sample_files)}, Labels: {len(label_files)}\n"

print(dataset_summary)
report.append(dataset_summary)


# Displaying samples
print("Loading and displaying samples")
print("-"*50)

def display_and_save_sample(sample_path, label_path, index, output_dir):
    # Load images
    img = Image.open(sample_path)
    label = Image.open(label_path)
    
    # Conversion
    label_array = np.array(label).astype(np.float32)
    label_array = (label_array / 255.0) * 2 - 1

    # Subplots and figure
    fig, axes = plt.subplots(1,2, figsize=(12,5))

    # Sample images
    axes[0].imshow(img)
    axes[0].set_title(f"Sample {index}\nSatellite Image", fontsize=12)
    axes[0].axis("off")

    # Labels 
    im = axes[1].imshow(label_array, cmap="viridis", vmin=-1, vmax=1)
    axes[1].set_title(f"Sample {index}\nNDVI Label", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='NDVI Value (-1 to 1)')

    plt.suptitle(f"Figure 1: Sample {index}: Satellite Image and NDVI Label", fontsize=14)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, f"sample_{index}_visualization.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()

    # Return image values
    return {
        'index': index,
        'img_size': img.size,
        'label_size': label.size,
        'label_data_type': label_array.dtype,
        'label_min': float(label_array.min()),
        'label_max': float(label_array.max()),
        'label_mean': float(label_array.mean()),
        'label_std': float(label_array.std()),
        'saved_fig': fig_path
    }

# Display first 3 samples
sample_info_array = []
for i in range(min(SAMPLE_TEST_AMOUNT, len(sample_files))):
    print(f"\nAnalyzing Sample {i}:")
    info = display_and_save_sample(sample_files[i], label_files[i], i, output_dir)
    sample_info_array.append(info)

    print(f"  Image size: {info['img_size']}")
    print(f"  Label size: {info['label_size']}")
    print(f"  NDVI range: [{info['label_min']:.3f}, {info['label_max']:.3f}]")
    print(f"  Mean NDVI: [{info['label_mean']:.3f}, standard: {info['label_std']:.3f}]")


# Data set characteristics
print("\nAnalyzing dataset characteristics")
print("="*50)

def analyze_dataset_properties(sample_files, label_files, num_samples=SAMPLE_TEST_AMOUNT):
    properties = {
        'image_sizes': [],
        'image_modes': [],
        'label_stats': [],
        'file_names': []
    }

    ndvi_values = []

    for i in range(min(num_samples, len(sample_files))):
        # load images
        img = Image.open(sample_files[i])
        label = Image.open(label_files[i])
        label_array = np.array(label)
        label_array = np.array(label_array / 255.0) * 2 - 1

        # Properties collection
        properties['image_sizes'].append(img.size)
        properties['image_modes'].append(img.mode)
        properties['file_names'].append(os.path.basename(sample_files[i]))

        # Collect NDVI values
        ndvi_values.extend(label_array.flatten())

        # Store label statistics
        properties['label_stats'].append({
            'min': float(label_array.min()),
            'max': float(label_array.max()),
            'mean': float(label_array.mean()),
            'std': float(label_array.std())
        })

    ndvi_values = np.array(ndvi_values)

    return properties, ndvi_values

# Run analysis
properties, all_ndvi = analyze_dataset_properties(sample_files, label_files)

# Compile characteristics
unique_sizes = set(properties['image_sizes'])
unique_modes = set(properties['image_modes'])

characteristics = f"""
DATASET CHARACTERISTICS
=======================
Image Format: TIFF
Image Sizes: {unique_sizes}
Image Modes: {unique_modes}

NDVI Analysis (based on {len(all_ndvi):,} pixels):
------------------------------------------------
Global Minimum : {all_ndvi.min():.3f}
Global Maximum: {all_ndvi.max():.3f}
Global Mean: {all_ndvi.mean():.3f}
Global Standard: {all_ndvi.std():.3f}

NDVI Distribution by category:
- Water/Cloud (NDVI < 0): {np.mean(all_ndvi < 0)*100:.1f}%
- Bare Soil (0 to 0.2): {np.mean((all_ndvi >= 0) & (all_ndvi < 0.2))*100:.1f}%
- Sparse Vegetation (0.2 to 0.5): {np.mean((all_ndvi >= 0.2) & (all_ndvi < 0.5))*100:.1f}%
- Dense Vegetation (n(number) >= 0.5): {np.mean(all_ndvi >= 0.5)*100:.1f}%
=======================
"""

print(characteristics)
report.append(characteristics)

# NDVI distribution plot
plt.subplot(1,2,1)
plt.hist(all_ndvi, bins=50, color='green', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='blue', linestyle='--', label='Bare soil threshold')
plt.title('NDVI Distribution Across Dataset')
plt.xlabel('NDVI Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
categories = ['Water/Clouds', 'Bare Soil', 'Sparse Veg', 'Dense Veg']
values = [
    np.mean(all_ndvi < 0)*100,
    np.mean((all_ndvi >= 0) & (all_ndvi < 0.2))*100,
    np.mean((all_ndvi >= 0.2) & (all_ndvi < 0.5))*100,
    np.mean(all_ndvi >= 0.5)*100
]
colors = ['blue', 'brown', 'yellowgreen', 'darkgreen']
plt.bar(categories, values, color=colors, alpha=0.7)
plt.title('Land Cover Distribution')
plt.xlabel('Category')
plt.ylabel('Percentage (%)')

for i, v in enumerate(values):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')

plt.tight_layout()
dist_plot_path = os.path.join(output_dir, "ndvi_distribution.png")
plt.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
plt.show()

print('Checking for Data Quality Issues')
print("="*50)

def check_data_quality(sample_files, label_files, num_samples=SAMPLE_TEST_AMOUNT):
    issues = []

    for i in range(min(num_samples, len(sample_files))):
        try:
            img = Image.open(sample_files[i])
            label = Image.open(label_files[i])

            img_array = np.array(img)

            # Conversion
            label_array = np.array(label).astype(np.float32)
            label_array = (label_array / 255.0) * 2 - 1 

            # Dimension mismatch check
            if img_array.shape[:2] != label_array.shape[:2]:
                issues.append(f"Size mismatch in pair {i}: {img_array.shape[:2]} vs {label_array.shape[:2]}")
            
            # Checking NaN values
            if np.isnan(img_array).any() or np.isnan(label_array).any():
                issues.append(f"NaN values found in {i}")

            # NVDI range validity
            if label_array.min() < -1.1 or label_array.max() > 1.1:
                issues.append(f"NDVI outside [-1,1] in {i}: [{label_array.min():.2f}, {label_array.max():.2f}]")

        except Exception as e:
            issues.append(f"Cannot open file {i}: {str(e)}")

    return issues

# Check for issues
quality_issues = check_data_quality(sample_files, label_files)

quality_report = f"""
DATA QUALITY ASSESSMENT
=======================
Files Checked: {min(SAMPLE_TEST_AMOUNT, len(sample_files))}
Issues Found: {len(quality_issues)}
=======================
"""

if quality_issues:
    quality_report += "Issues Detected:\n"
    for issue in quality_issues[:10]:
        quality_report += f"  - {issue}\n"
    if len(quality_issues) > 10:
        quality_report += f"  ... and {len(quality_issues) - 10} more issues\n"

else:
    quality_report += "No data quality issues found"

print(quality_report)
report.append(quality_report)

# Preprocessing demonstartions
print("Preprocessing Demonstrations")
print("="*50)

def demonstrate_preprocessing(sample_path, label_path, output_dir):
    # Load original images
    img = Image.open(sample_path)
    label = Image.open(label_path)

    # Original arrays
    img_array = np.array(img)
    label_array = np.array(label)

    # Resizing
    target_size = (128, 128)
    img_resized = img.resize(target_size)
    label_resized = label.resize(target_size, Image.NEAREST)

    # Normalization
    img_norm = np.array(img_resized).astype(np.float32)
    if img_norm.max() > 1.0:
        img_norm = img_norm / 255.0

    # NDVI scaling
    label_norm = np.array(label_resized).astype(np.float32)
    if label_norm.max() > 1.0:
        label_norm = (label_norm / 255.0) * 2 - 1

    # Visuals
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Images
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"Original\n{img.size}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_resized)
    axes[0, 1].set_title(f"Resized\n{target_size}")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_norm)
    axes[0, 2].set_title(f"Normalized\n[{img_norm.min():.2f}, {img_norm.max():.2f}]")
    axes[0, 2].axis('off')
    
    axes[0, 3].axis('off')  
    
    # Labels
    axes[1, 0].imshow(label, cmap='viridis')
    axes[1, 0].set_title(f"Original Label\n{label.size}")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(label_resized, cmap='viridis')
    axes[1, 1].set_title(f"Resized Label")
    axes[1, 1].axis('off')
    
    im = axes[1, 2].imshow(label_norm, cmap='viridis', vmin=-1, vmax=1)
    axes[1, 2].set_title(f"Normalized NDVI\n[{label_norm.min():.2f}, {label_norm.max():.2f}]")
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])
    
    axes[1, 3].axis('off')
    
    plt.suptitle("Figure 3: Preprocessing Pipeline Demonstration", fontsize=14)
    plt.tight_layout()
    
    # Save figure
    prep_path = os.path.join(output_dir, "preprocessing_demo.png")
    plt.savefig(prep_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'original_size': img.size,
        'target_size': target_size,
        'image_range': (float(img_norm.min()), float(img_norm.max())),
        'ndvi_range': (float(label_norm.min()), float(label_norm.max()))
    }


# Show preprocessing on first sample
prep_info = demonstrate_preprocessing(sample_files[0], label_files[0], output_dir)

preprocessing_summary = f"""
PREPROCESSING DEMONSTRATION
===========================
Original image size: {prep_info['original_size']}
Resized to: {prep_info['target_size']}
Normalized image range: {prep_info['image_range']}
Normalized NDVI range: {prep_info['ndvi_range']}

Recommended preprocessing pipeline:
1. Resize all images to consistent dimensions (e.g., 128x128 or 256x256)
2. Normalize image pixel values to [0, 1] range by dividing by 255
3. Ensure NDVI values are in [-1, 1] range (convert if stored as 0-255)
4. Use NEAREST neighbor interpolation for labels to preserve class values
5. Consider data augmentation for training (flips, rotations, etc.)
===========================
"""

print(preprocessing_summary)
report.append(preprocessing_summary)

print("Generating Final Report")
print("="*50)

# Compile final report
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
final_report = f"""
===============================================================================
REMOTE SENSING VEGETATION DATASET ANALYSIS REPORT
===============================================================================

Report Generated: {timestamp}
Analysis Script: dataset_analysis.py
Output Directory: {output_dir}

{report[0]}  # Dataset Overview

{report[1]}  # Dataset Characteristics

{report[2]}  # Data Quality

{report[3]}  # Preprocessing Demonstration



===============================================================================
VISUALIZATIONS SAVED
===============================================================================
The following visualization files have been saved to {output_dir}:
1. sample_*_visualization.png - Sample images with their NDVI labels
2. ndvi_distribution.png - NDVI distribution histogram and land cover breakdown
3. preprocessing_demo.png - Preprocessing pipeline demonstration

===============================================================================
CONCLUSIONS AND RECOMMENDATIONS
===============================================================================

Based on this analysis, the dataset is {'READY' if len(quality_issues) == 0 else 'IN NEED OF REVIEW'} 
for machine learning experiments.

Key Findings:
- Total samples: {len(sample_files)} with matching labels
- Image format: TIFF, {unique_sizes}
- NDVI range: [{all_ndvi.min():.3f}, {all_ndvi.max():.3f}]
- Data quality: {len(quality_issues)} issues detected
- Landscape type: {np.mean(all_ndvi >= 0.5)*100:.1f}% dense vegetation, {np.mean(all_ndvi < 0.2)*100:.1f}% non-vegetated

Next Steps:
1. {'Address identified data issues before proceeding' if len(quality_issues) > 0 else 'Proceed with model development'}
2. Implement preprocessing pipeline as demonstrated
3. Split data into train/validation/test sets
4. Begin model experimentation 

===============================================================================
END OF REPORT
===============================================================================
"""

# Save report to text file
report_path = os.path.join(output_dir, "dataset_analysis_report.txt")
with open(report_path, 'w') as f:
    f.write(final_report)

print(f"\nFinal report saved to: {report_path}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Display final report preview
print("\nFinal Report Preview:")
print("="*50)
print(final_report[:1000] + "...\n[Report truncated for display]")
print(f"\nFull report available at: {report_path}")

print(len(report))