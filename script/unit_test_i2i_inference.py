import autorootcwd
import glob
import os
import lightning as L
from src.register_by_gen.dl_model.model_i2i import I2IModel
from src.register_by_gen.dataset.syn2023_dset import SynthRAD2023DataModule

def test_i2i_inference():
    """Test I2I model using Lightning trainer.test() method."""
    
    print("Testing I2I Model with Lightning Trainer...")
    
    # 1. Setup DataModule
    print("\nSetting up DataModule...")
    data_module = SynthRAD2023DataModule(
        mode='2d', 
        batch_size=1  # Use batch_size=1 for testing
    )
    print("DataModule created")
    
    # 2. Create Lightning Trainer
    print("\nCreating Lightning Trainer...")
    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        logger=False  # Disable logging for testing
    )
    print("Trainer created")
    
    # 3. Load model from checkpoint
    print("\nLoading model from checkpoint...")
    
    # Find latest checkpoint automatically
    checkpoint_dir = 'logs/i2i_experiment/version_1/checkpoints/'
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        print(f"No checkpoint files found in: {checkpoint_dir}")
        return
    
    # Use the latest checkpoint (sorted by filename)
    checkpoint_path = sorted(checkpoint_files)[-1]
    print(f"Found checkpoint: {os.path.basename(checkpoint_path)}")
    
    try:
        model = I2IModel.load_from_checkpoint(checkpoint_path)
        print("Model loaded from checkpoint")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
    
    # 4. Run test with loaded model
    print("\nRunning test...")
    results = trainer.test(
        model=model,
        datamodule=data_module,
        verbose=True
    )
    
    print("\nTest completed!")
    print(f"Results: {results}")

if __name__ == "__main__":
    test_i2i_inference()