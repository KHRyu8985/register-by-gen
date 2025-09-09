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

This project uses DVC (Data Version Control) for managing large medical image datasets.

### Data Management with DVC

#### First-time Setup
```bash
# Pull data from remote storage
dvc pull
```

#### Current Dataset Structure
```
data/
├── syn23_pelvic/              # SynthRAD2023 Pelvic Dataset (11GB+)
│   ├── train/                 # Training data (130 subjects)
│   │   ├── 1PA001/           # Patient ID (Pelvis T2 → CT)
│   │   │   ├── ct.nii.gz     # Reference CT image
│   │   │   ├── mr.nii.gz     # Input MR T2 image  
│   │   │   └── mask.nii.gz   # Body mask
│   │   ├── 1PA010/
│   │   ├── 1PC000/           # Patient ID (Pelvis T1 → CT)
│   │   └── ... (130 total)
│   ├── test/                 # Test data (33 subjects)
│   │   ├── 1PA004/
│   │   ├── 1PC006/
│   │   └── ... (33 total)
│   ├── val/                  # Validation data (17 subjects)
│   │   ├── 1PA005/
│   │   ├── 1PC007/
│   │   └── ... (17 total)
│   └── overview/             # Dataset overview images
└── test_ct_cbct/             # Legacy CT-CBCT test data
    └── AB_D/
        ├── 2ABD001/
        │   ├── ct.mha        # CT image (legacy format)
        │   ├── cbct.mha      # CBCT image  
        │   └── mask.mha      # Segmentation mask
        └── 2ABD003/
```

#### Dataset Details
- **Total Subjects**: 180 (130 train + 33 test + 17 val)
- **Image Types**: 
  - **MR → CT**: T1-weighted and T2-weighted MR to CT synthesis
  - **Patient IDs**: 1PA### (T2→CT), 1PC### (T1→CT)
- **File Format**: NIfTI (.nii.gz) - Standard medical imaging format
- **Image Dimensions**: Varies by patient, typically ~512×512×(100-200) voxels
- **Voxel Spacing**: ~1mm isotropic resolution

#### Working with DVC (Optional)
```bash
# Check data status
dvc status

# Pull latest data version
dvc pull

# After modifying data, push changes
dvc add data/
git add data.dvc .gitignore
git commit -m "Update dataset"
dvc push

# Checkout specific data version
git checkout <commit-hash>
dvc pull
```

#### DVC Remote Configuration
- **Remote storage**: `/data/kanghyun/register-by-gen`
- **Total files**: 728 files backed up
- **Data tracking**: Managed via `data.dvc` file

### Data Description
- **CT**: High-resolution computed tomography images
- **CBCT**: Cone beam CT images (lower resolution, with artifacts)
- **Mask**: Region of interest segmentation masks
- **Format**: Medical image format (.mha files)

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

### TorchIO-based Dataset Implementation (`src/register_by_gen/dataset/`)

#### SynthRAD2023 Dataset Class
Unified dataset implementation for handling CT/MR pairs from the SynthRAD2023 pelvic dataset:

- **File**: `syn2023_dset.py` - Core dataset class with TorchIO pipeline
- **Lightning DataModule**: `lightning_datamodule.py` - PyTorch Lightning compatible wrapper
- **Total Subjects**: 130 training subjects (CT/MR pairs)
- **Data Format**: NIfTI files (ct.nii.gz, mr.nii.gz)

#### Multi-modal Training Support
```python
# 2D mode: Single slices (batch_size=16)
loader_2d = create_train_loader(mode='2d')
# Output: (batch_size, 1, 128, 128)

# 2.5D mode: 3 adjacent slices (batch_size=12)  
loader_25d = create_train_loader(mode='2.5d')
# Output: (batch_size, 3, 128, 128)

# 3D mode: Full volume patches (batch_size=1)
loader_3d = create_train_loader(mode='3d')
# Output: (batch_size, 1, 96, 96, 64)
```

#### Key Features:
- **Preprocessing Pipeline**: RescaleIntensity, ZNormalization, ToCanonical
- **Data Augmentation**: RandomFlip, RandomAffine transformations
- **Queue System**: Efficient patch generation with TorchIO Queue
- **Custom Patch Sizes**: Configurable patch dimensions for different modes

## Testing Scripts (`script/`)

### DataModule Testing (`script/test_lightning_datamodule.py`)
Comprehensive testing script for the Lightning DataModule implementation:

#### Features Tested:
- **Multi-mode Support**: Tests 2D, 2.5D, and 3D training modes
- **Batch Shape Validation**: Verifies correct tensor dimensions for each mode
- **Custom Patch Sizes**: Tests configurable patch dimensions
- **Error Handling**: Validates input parameters and catches invalid configurations

#### Usage:
```bash
uv run python script/test_lightning_datamodule.py
```

#### Sample Output:
```
=== Testing Lightning DataModule ===

--- 2D DataModule ---
Train batch - IMG: torch.Size([8, 1, 128, 128]), REF: torch.Size([8, 1, 128, 128])

--- 2.5D DataModule ---
2.5D batch - IMG: torch.Size([6, 3, 128, 128]), REF: torch.Size([6, 3, 128, 128])

--- Mode Comparison ---
2d mode - IMG: torch.Size([16, 1, 128, 128]), REF: torch.Size([16, 1, 128, 128])
2.5d mode - IMG: torch.Size([12, 3, 128, 128]), REF: torch.Size([12, 3, 128, 128])
3d mode - IMG: torch.Size([1, 1, 96, 96, 64]), REF: torch.Size([1, 1, 96, 96, 64])
```

### I2I Model Training Pipeline (`script/test_i2i_pipeline.py`)
Complete end-to-end training pipeline test with PyTorch Lightning:

#### Features:
- **Lightning Integration**: Uses Lightning Trainer for training
- **I2I Model**: Tests Image-to-Image translation model
- **CSV Logging**: Automatic experiment logging
- **GPU Support**: Auto-detection of available devices
- **Multi-Device Support**: Configured for distributed training

#### Configuration:
- **Model**: I2IModel (1 input channel → 1 output channel)
- **Learning Rate**: 1e-3
- **Training Mode**: 2D with batch_size=12
- **Max Epochs**: 100
- **Logger**: CSV format saved to `logs/`

#### Usage:
```bash
uv run python script/test_i2i_pipeline.py
```

#### Pipeline Steps:
1. **DataModule Setup**: Creates and configures SynthRAD2023DataModule
2. **Model Creation**: Initializes I2I Lightning model
3. **Logger Setup**: Configures CSV logging for experiment tracking
4. **Trainer Configuration**: Sets up Lightning Trainer with GPU support
5. **Training Execution**: Runs complete training loop

## Project Structure

```
register-by-gen/
├── .gitignore              # Git ignore file settings
├── README.md              # Project documentation  
├── pyproject.toml         # Python project configuration
├── uv.lock               # uv dependency lock file
├── data/                 # Dataset directory (managed by DVC)
│   ├── syn23_pelvic/     # SynthRAD2023 pelvic dataset
│   └── test_ct_cbct/     # Additional test data
├── nbs/                  # Jupyter notebooks
│   ├── dataset.ipynb     # TorchIO dataset tutorial
│   └── torchio_dataset.ipynb  # Additional TorchIO examples
├── script/               # Test and utility scripts
│   ├── test_lightning_datamodule.py  # DataModule testing
│   └── test_i2i_pipeline.py         # I2I training pipeline test
├── pred_ct_vols/         # Prediction results storage
├── logs/                 # Training logs and experiments
└── src/                  # Source code
    └── register_by_gen/
        ├── dataset/      # Dataset implementations
        │   ├── syn2023_dset.py      # TorchIO dataset class
        │   └── lightning_datamodule.py  # Lightning wrapper
        └── dl_model/     # Deep learning models
            ├── model_i2i.py         # Image-to-Image model
            ├── model_gen.py         # Generator model
            └── model_reg.py         # Registration model
```