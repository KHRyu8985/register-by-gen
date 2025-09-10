"""
Test script for I2I model with SynthRAD2023 dataset using Lightning DataModule.
"""

import autorootcwd
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from src.register_by_gen.dl_model.model_i2i import I2IModel
from src.register_by_gen.dataset.syn2023_dset import SynthRAD2023DataModule
import os
os.environ["NCCL_P2P_DISABLE"] = "1" # remove error of ddp
torch.set_float32_matmul_precision('medium')

def test_i2i_pipeline():
    """Test complete I2I training pipeline with Lightning DataModule."""
    
    print("🚀 Testing I2I Pipeline with Lightning DataModule...")
    
    # 1. Create DataModule
    print("\n📊 Creating DataModule...")
    data_module = SynthRAD2023DataModule(
        mode='2d',
        batch_size=12
    )
    data_module.setup('fit')
    print("✅ DataModule created and setup completed")
    
    # 2. Create model
    print("\n🧠 Creating I2I Lightning model...")
    model = I2IModel(
        in_channels=1,
        out_channels=1,
        lr=1e-3
    )
    print("✅ I2I model created")
    
    # 3. Create CSV Logger
    print("\n📋 Creating CSV Logger...")
    csv_logger = CSVLogger(save_dir="logs", name="i2i_experiment")
    print("✅ CSV Logger created - logs will be saved to logs/")
    
    # 4. Create Lightning Trainer
    print("\n⚡ Creating Lightning Trainer...")
    trainer = L.Trainer(
        max_epochs=5,
        enable_checkpointing=True,
        logger=csv_logger,
        enable_progress_bar=True,
        accelerator='auto',
        devices=1,
        num_sanity_val_steps=1  # Skip validation sanity check
    )
    print("✅ Trainer created")
    
    # 5. Test training
    print("\n🏋️ Testing training steps...")
    trainer.fit(model, datamodule=data_module)
    print("✅ Training completed successfully!")
    trainer.test(model, datamodule=data_module)
    print("✅ Testing completed successfully!")

if __name__ == "__main__":
    test_i2i_pipeline()