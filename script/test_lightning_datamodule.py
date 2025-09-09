"""
Test script for Lightning DataModule.
"""

import autorootcwd
from src.register_by_gen.dataset import SynthRAD2023DataModule


def main():
    print("=== Testing Lightning DataModule ===")
    
    # Test 2D mode
    print("\n--- 2D DataModule ---")
    dm = SynthRAD2023DataModule(mode='2d', batch_size=8)
    dm.setup()
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    train_batch = next(iter(train_loader))
    print(f"Train batch - IMG: {train_batch['img'].shape}, REF: {train_batch['ref'].shape}")
    
    val_batch = next(iter(val_loader))
    print(f"Val batch - IMG: {val_batch['img'].shape}, REF: {val_batch['ref'].shape}")
    
    # Test 2.5D mode
    print("\n--- 2.5D DataModule ---")
    dm_25d = SynthRAD2023DataModule(mode='2.5d', batch_size=6)
    dm_25d.setup()
    
    train_loader_25d = dm_25d.train_dataloader()
    batch_25d = next(iter(train_loader_25d))
    print(f"2.5D batch - IMG: {batch_25d['img'].shape}, REF: {batch_25d['ref'].shape}")
    
    # Test all modes
    print("\n--- Mode Comparison ---")
    modes = ['2d', '2.5d', '3d']
    
    for mode in modes:
        dm = SynthRAD2023DataModule(mode=mode)
        dm.setup()
        
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        if mode == '3d':
            import torchio as tio
            img_shape = batch['img'][tio.DATA].shape
            ref_shape = batch['ref'][tio.DATA].shape
        else:
            img_shape = batch['img'].shape
            ref_shape = batch['ref'].shape
            
        print(f"{mode} mode - IMG: {img_shape}, REF: {ref_shape}")
    
    # Test custom patch sizes
    print("\n--- Custom Patch Sizes ---")
    
    # 2D with custom patch size
    dm_custom_2d = SynthRAD2023DataModule(mode='2d', patch_size=(64, 64), batch_size=4)
    dm_custom_2d.setup()
    batch = next(iter(dm_custom_2d.train_dataloader()))
    print(f"2D custom (64,64) - IMG: {batch['img'].shape}")
    
    # 3D with custom patch size  
    dm_custom_3d = SynthRAD2023DataModule(mode='3d', patch_size=(64, 64, 32), batch_size=1)
    dm_custom_3d.setup()
    batch = next(iter(dm_custom_3d.train_dataloader()))
    print(f"3D custom (64,64,32) - IMG: {batch['img'][list(batch['img'].keys())[0]].shape}")
    
    # Test validation error
    print("\n--- Testing Validation ---")
    try:
        dm_invalid = SynthRAD2023DataModule(mode='2d', patch_size=(64, 64, 32))  # Wrong dimensions
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\n=== DataModule Test Complete ===")


if __name__ == "__main__":
    main()