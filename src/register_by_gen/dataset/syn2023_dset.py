import autorootcwd
import os, glob, torch
import torchio as tio
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

if __name__ == "__main__":
    import time

    def peek(loader, name, n_batches=3):
        print(f"\n=== {name} ===")
        t0 = time.time()
        it = iter(loader)
        b0 = next(it)
        print("keys:", list(b0.keys()))
        print("img:", tuple(b0["img"].shape), "ref:", tuple(b0["ref"].shape))
        print("subject_id[0]:", b0["subject_id"][0] if isinstance(b0["subject_id"], (list, tuple)) else b0["subject_id"])
        print(f"first batch time: {time.time()-t0:.3f}s")
        # run a few more to see stability
        c, t1 = 1, time.time()
        with torch.no_grad():
            for _ in range(n_batches-1):
                b = next(it)
                _ = b["img"], b["ref"]
                c += 1
        print(f"{c} batches ok, elapsed {time.time()-t1:.3f}s")

    # 데이터 루트 경로 필요시 환경에 맞게 수정
    data_root = "data/syn23_pelvic"

    # train loaders
    dset_train = SynthRad2023Dataset(data_root, split="train")
    peek(dset_train.get_2d_train_loader(patch_size=(128,128,1), batch_size=4, samples_per_volume=4),
         "Train 2D")

    # test loader (full volume)
    dset_test = SynthRad2023Dataset(data_root, split="test")
    peek(dset_test.get_test_loader(batch_size=1), "Test Full Volume", n_batches=1)

    print("\nAll loaders produced tensor-only batches (img/ref/subject_id). ✅")
