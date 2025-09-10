import autorootcwd
import os, glob, torch
import torchio as tio
import lightning as L
from torch.utils.data import DataLoader

class SynthRad2023Dataset:
    def __init__(self, data_root: str, split: str = 'train'):
        self.data_root, self.split = data_root, split
        self.subjects = self._load_subjects()
        print(f"Loaded {len(self.subjects)} subjects for {split} split")

    def _load_subjects(self):
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        subs = []
        for sid in sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))):
            sd = os.path.join(split_dir, sid)
            ct, mr = glob.glob(os.path.join(sd, 'ct.nii*')), glob.glob(os.path.join(sd, 'mr.nii*'))
            if ct and mr:
                subs.append(tio.Subject(ref=tio.ScalarImage(ct[0]), img=tio.ScalarImage(mr[0]), subject_id=sid))
        if not subs: raise ValueError(f"No subjects with ct.nii*/mr.nii* in {split_dir}")
        return subs

    @staticmethod
    def _tfm_train():
        return tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(percentiles=(0.5,99.5), include=['ref','img']),
            tio.ZNormalization(include=['ref','img']),
            tio.RandomFlip(axes=('LR',), include=['ref','img']),
            tio.RandomAffine(scales=(0.95,1.05), degrees=(0,0,10), translation=(5,5,0), include=['ref','img']),
        ])

    @staticmethod
    def _tfm_eval():
        return tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(percentiles=(0.5,99.5), include=['ref','img']),
            tio.ZNormalization(include=['ref','img']),
        ])

    @staticmethod
    def _queue(subjects, tfm, patch_size, spv, max_q, nw):
        ds = tio.SubjectsDataset(subjects, transform=tfm)
        return tio.Queue(ds, max_length=max_q, samples_per_volume=spv,
                         sampler=tio.UniformSampler(patch_size),
                         num_workers=nw, shuffle_subjects=True, shuffle_patches=True)

    # -------- tensor-only wrappers (img/ref/subject_id only) --------
    @staticmethod
    def _wrap_3d(q):
        class DS:
            def __len__(self): return len(q)
            def __getitem__(self, i):
                p = q[i]
                return {'img': p['img'][tio.DATA].contiguous().float(),
                        'ref': p['ref'][tio.DATA].contiguous().float(),
                        'subject_id': p.get('subject_id','unknown')}
        return DS()

    @staticmethod
    def _wrap_2d(q):  # (1,H,W,1) -> (1,H,W)
        class DS:
            def __len__(self): return len(q)
            def __getitem__(self, i):
                p = q[i]
                img = p['img'][tio.DATA].squeeze(-1).contiguous().float()
                ref = p['ref'][tio.DATA].squeeze(-1).contiguous().float()
                return {'img': img, 'ref': ref, 'subject_id': p.get('subject_id','unknown')}
        return DS()

    @staticmethod
    def _wrap_25d(q):  # (1,H,W,3) -> (3,H,W)
        class DS:
            def __len__(self): return len(q)
            def __getitem__(self, i):
                p = q[i]
                img = p['img'][tio.DATA].squeeze(0).permute(2,0,1).contiguous().float()
                ref = p['ref'][tio.DATA].squeeze(0).permute(2,0,1).contiguous().float()
                return {'img': img, 'ref': ref, 'subject_id': p.get('subject_id','unknown')}
        return DS()

    # ---------------- public loaders ----------------
    def get_train_loader(self, patch_size=(96,96,48), batch_size=8,
                         samples_per_volume=10, max_queue_length=200,
                         num_workers=0, pin_memory=False):
        ds = self._wrap_3d(self._queue(self.subjects, self._tfm_train(),
                                       patch_size, samples_per_volume, max_queue_length, num_workers))
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    def get_2d_train_loader(self, patch_size=(128,128,1), batch_size=16,
                            samples_per_volume=20, max_queue_length=200,
                            num_workers=8, pin_memory=False):
        ds = self._wrap_2d(self._queue(self.subjects, self._tfm_train(),
                                       patch_size, samples_per_volume, max_queue_length, num_workers))
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    def get_25d_train_loader(self, patch_size=(128,128,3), batch_size=12,
                             samples_per_volume=20, max_queue_length=200,
                             num_workers=8, pin_memory=False):
        ds = self._wrap_25d(self._queue(self.subjects, self._tfm_train(),
                                        patch_size, samples_per_volume, max_queue_length, num_workers))
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    def get_test_loader(self, batch_size=1, pin_memory=False):
        subjects, tfm = self.subjects, self._tfm_eval()
        class Full:
            def __len__(self): return len(subjects)
            def __getitem__(self, i):
                s = tfm(subjects[i])
                return {'img': s['img'][tio.DATA].contiguous().float(),
                        'ref': s['ref'][tio.DATA].contiguous().float(),
                        'subject_id': s.get('subject_id', f'subject_{i}')}
        return DataLoader(Full(), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

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
        patch_size: tuple = None, # (192,192), (192,192), (96,96,48)
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
