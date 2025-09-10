"""
Training script using Lightning CLI for configurable parameters.
Allows easy modification of model and training parameters via command line or config files.
"""

import autorootcwd
import os
import torch
from lightning.pytorch.cli import LightningCLI
from src.register_by_gen.dl_model.model_i2i import I2IModel
from src.register_by_gen.dataset.syn2023_dset import SynthRAD2023DataModule

# Set environment variables for better performance
os.environ["NCCL_P2P_DISABLE"] = "1"
torch.set_float32_matmul_precision('medium')


class I2ICLI(LightningCLI):
    """Custom Lightning CLI for I2I training with enhanced configuration."""
    
    def add_arguments_to_parser(self, parser):
        """Add custom arguments to the CLI parser."""
        
        # Add model-specific arguments
        parser.add_lightning_class_args(I2IModel, "model")
        parser.add_lightning_class_args(SynthRAD2023DataModule, "data")
        
        # Add custom training arguments
        parser.add_argument("--experiment_name", type=str, default="i2i_cli_experiment",
                          help="Name of the experiment for logging")
        parser.add_argument("--log_dir", type=str, default="logs",
                          help="Directory to save logs")
        parser.add_argument("--seed", type=int, default=42,
                          help="Random seed for reproducibility")


def cli_main():
    """Main CLI function."""
    
    # Create CLI with custom configuration
    cli = I2ICLI(
        model_class=I2IModel,
        datamodule_class=SynthRAD2023DataModule,
        seed_everything_default=42,
        trainer_defaults={
            "max_epochs": 50,
            "accelerator": "auto",
            "devices": 1,
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 1,
            "enable_progress_bar": True,
        },
        save_config_callback=None,  # Disable automatic config saving
    )


if __name__ == "__main__":
    cli_main()