# Running the Python Scripts

This repository contains multiple Python scripts for machine learning and data analysis. Follow the instructions below to set up your environment and execute the scripts.

## Prerequisites
Ensure you have Python installed on your system (Python 3.8 or later recommended). You can check your Python version by running:
```bash
python --version
```

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Scripts

### 1. Running `testone.py`
This script performs a linear regression analysis on the California housing dataset.
```bash
python testone.py
```

### 2. Running `test2partone.py` and `test2parttwo.py`
These scripts load and process the UCI HAR dataset for human activity recognition.

**Ensure the dataset is available in the correct directory (`UCI HAR Dataset/`).**

Run the scripts sequentially:
```bash
python test2partone.py
python test2parttwo.py
```

**Note:** `test2partone.py` focuses on data preprocessing and PCA analysis, while `test2parttwo.py` includes subject-wise data filtering and classification.

## Notes
- `testone.py` performs a linear regression analysis on the California housing dataset.
- `test2partone.py` preprocesses the UCI HAR dataset and applies PCA.
- `test2parttwo.py` applies subject-wise filtering and trains an SVM classifier.
- Make sure all required datasets are available in the appropriate directories.

## Troubleshooting
If you encounter issues:
- Ensure all dependencies are installed (`pip install -r requirements.txt`).
- Verify that the required datasets are in place.
- Check for errors and debug as needed.



