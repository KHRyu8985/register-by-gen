# 3D Medical Image Registration by Generation

A research project for performing CT-CBCT registration using generative models with 3D medical image data.

## Overview

This project aims to improve medical image analysis accuracy through precise registration between CT (Computed Tomography) and CBCT (Cone Beam CT) images.

## Requirements

- Python 3.12
- PyTorch 2.7.1+cu128
- CUDA 12.8
- uv (Python package manager)

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd register-by-gen
```

2. **Setup uv environment**
```bash
# Install dependencies
uv sync

# Install NATTEN library (위에서 알아서 설치가 안되면면)
uv add "natten @ https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.0/natten-0.21.0%2Btorch270cu128-cp312-cp312-linux_x86_64.whl"
```

3. **Setup Jupyter kernel**
```bash
uv add ipykernel jupyter
uv run python -m ipykernel install --user --name register-by-gen
```

## Dataset

### Structure
```
data/Train_D/AB_D/
├── 2ABD001/
│   ├── ct.mha      # CT image
│   ├── cbct.mha    # CBCT image
│   └── mask.mha    # Segmentation mask
└── 2ABD003/
    ├── ct.mha
    ├── cbct.mha
    └── mask.mha
```

### Data Description
- **CT**: High-resolution computed tomography images
- **CBCT**: Cone beam CT images (lower resolution, with artifacts)
- **Mask**: Region of interest segmentation masks

## Tutorial

### Dataset Loading and Preprocessing (`nbs/dataset.ipynb`)

This notebook provides a complete pipeline for 3D medical image processing using TorchIO:

#### Key Features:
- **3D Medical Image Loading**: Loading CT/CBCT data with TorchIO
- **Data Preprocessing**:
  - `RescaleIntensity`: Percentile-based intensity normalization
  - `ZNormalization`: Z-score normalization
  - `ToCanonical`: Anatomical alignment
- **Data Augmentation**:
  - `RandomFlip`: Random flipping
  - `RandomAffine`: Random affine transformations
- **2D Patch Sampling**: Extracting 2D slices from 3D volumes
- **Sliding Window Inference**: Full volume processing
- **Result Saving**: Saving in medical image format (.mha)

#### Usage Example:
```python
import torchio as tio

# Create subject
subject = tio.Subject(
    ct=tio.ScalarImage('data/Train_D/AB_D/2ABD001/ct.mha'),
    cbct=tio.ScalarImage('data/Train_D/AB_D/2ABD001/cbct.mha'),
    mask=tio.LabelMap('data/Train_D/AB_D/2ABD001/mask.mha')
)

# Visualization
subject.plot()

# Preprocessing pipeline
transform = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99)),
    tio.ZNormalization(),
    tio.ToCanonical()
])

transformed = transform(subject)
```

## Usage

### Running Jupyter Notebooks
```bash
# Start Jupyter Lab
uv run jupyter lab

# Or Jupyter Notebook
uv run jupyter notebook
```

### Running Python Scripts
```bash
uv run python script.py
```

## Project Structure

```
register-by-gen/
├── .gitignore              # Git ignore file settings
├── README.md              # Project documentation
├── pyproject.toml         # Python project configuration
├── uv.lock               # uv dependency lock file
├── data/                 # Dataset directory
│   └── Train_D/AB_D/     # Training data
├── nbs/                  # Jupyter notebooks
│   └── dataset.ipynb     # Dataset tutorial
├── pred_ct_vols/         # Prediction results storage
└── src/                  # Source code
    └── register_by_gen/
```