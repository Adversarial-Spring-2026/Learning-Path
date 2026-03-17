# Dataset Analysis Script - Pandahat Adversarial

This script analyzes the remote sensing vegetation dataset (614 satellite images with NDVI labels) to understand dataset structure, quality, and preprocessing needs.

## Purpose

- Load and visualize sample images with their NDVI labels
- Analyze dataset characteristics (image format, resolution, NDVI distribution)
- Check for data quality issues
- Demonstrate preprocessing steps for machine learning

## Files

- `dataset_analysis.py` - Main analysis script
- `requirements.txt` - Python dependencies
- `analysis_output/` - Generated results (created when run)

## Usage

### 1. Set up virtual environment (recommended if you haven't already)
```bash
# Create virtual environment
python -m venv .venv
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
cd data_exploration
python dataset_analysis.py
```

### 4. View results
Check the `analysis_output/` folder for:
- Sample visualizations (`sample_*_visualization.png`)
- NDVI distribution plot (`ndvi_distribution.png`)
- Preprocessing demo (`preprocessing_demo.png`)
- Full report (`dataset_analysis_report.txt`)

## Configuration

Edit `SAMPLE_TEST_AMOUNT` at the top of the script to analyze more/fewer samples:
```python
SAMPLE_TEST_AMOUNT = 5  # Change this value (default: 5)
```

## Notes

- This script is designed to work with the existing folder structure in the main repo
- Results are saved to `analysis_output/` - delete this folder to rerun fresh
-  The analysis report (`dataset_analysis_report.txt`) is generated automatically when the script runs

## Dependencies

- numpy
- matplotlib
- Pillow

See `requirements.txt` for exact versions.