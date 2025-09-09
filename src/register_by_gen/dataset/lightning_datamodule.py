"""
PyTorch Lightning DataModule for SynthRAD2023 dataset.
Simple and clean interface for medical image registration.
"""

import lightning as L
from torch.utils.data import DataLoader
from .syn2023_dset import SynthRad2023Dataset


class SynthRAD2023DataModule(L.LightningDataModule):
    """
    Lightning DataModule for SynthRAD2023 pelvic dataset.
    
    Simple usage:
        dm = SynthRAD2023DataModule(mode='2d')
        dm.setup()
        train_loader = dm.train_dataloader()
    """
    
    def __init__(
        self,
        data_root: str = 'data/syn23_pelvic',
        mode: str = '2d',  # '2d', '2.5d', '3d'
        patch_size: tuple = None,
        batch_size: int = None,
        num_workers: int = 0,
        samples_per_volume: int = 20,
    ):
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        self.num_workers = num_workers
        self.samples_per_volume = samples_per_volume
        
        # Auto patch size based on mode
        if patch_size is None:
            self.patch_size = {
                '2d': (128, 128),
                '2.5d': (128, 128), 
                '3d': (96, 96, 48)
            }[mode]
        else:
            self.patch_size = patch_size
            self._validate_patch_size()
        
        # Auto batch size based on mode
        if batch_size is None:
            self.batch_size = {
                '2d': 16,
                '2.5d': 12, 
                '3d': 1
            }[mode]
        else:
            self.batch_size = batch_size
        
        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _validate_patch_size(self):
        """Validate patch_size matches the mode."""
        if self.mode in ['2d', '2.5d'] and len(self.patch_size) != 2:
            raise ValueError(f"For {self.mode} mode, patch_size must be 2D tuple (H, W), got {self.patch_size}")
        elif self.mode == '3d' and len(self.patch_size) != 3:
            raise ValueError(f"For 3d mode, patch_size must be 3D tuple (H, W, D), got {self.patch_size}")
        
        # Check values are positive
        if any(x <= 0 for x in self.patch_size):
            raise ValueError(f"All patch_size values must be positive, got {self.patch_size}")
    
    def setup(self, stage: str = None):
        """Setup datasets for train/val/test."""
        if stage == 'fit' or stage is None:
            self.train_dataset = SynthRad2023Dataset(self.data_root, split='train')
            self.val_dataset = SynthRad2023Dataset(self.data_root, split='val')
        
        if stage == 'test' or stage is None:
            self.test_dataset = SynthRad2023Dataset(self.data_root, split='test')
    
    def train_dataloader(self):
        """Create training dataloader based on mode."""
        if self.mode == '2d':
            patch_size_3d = (*self.patch_size, 1)  # (H, W) -> (H, W, 1)
            return self.train_dataset.get_2d_train_loader(
                patch_size=patch_size_3d,
                batch_size=self.batch_size,
                samples_per_volume=self.samples_per_volume
            )
        elif self.mode == '2.5d':
            patch_size_3d = (*self.patch_size, 3)  # (H, W) -> (H, W, 3)
            return self.train_dataset.get_25d_train_loader(
                patch_size=patch_size_3d,
                batch_size=self.batch_size,
                samples_per_volume=self.samples_per_volume
            )
        else:  # 3d
            return self.train_dataset.get_train_loader(
                patch_size=self.patch_size,
                batch_size=self.batch_size,
                samples_per_volume=10  # 3D needs fewer samples
            )
    
    def val_dataloader(self):
        """Create validation dataloader (full volumes)."""
        return self.val_dataset.get_test_loader(batch_size=1)
    
    def test_dataloader(self):
        """Create test dataloader (full volumes).""" 
        return self.test_dataset.get_test_loader(batch_size=1)
    
    def predict_dataloader(self):
        """Same as test dataloader for inference."""
        return self.test_dataloader()

