"""
MONAI-based image-to-image translation with PyTorch Lightning.
Simple L1 loss only version.
"""

import torch
import torch.nn as nn
import lightning as L
import torchio as tio
from monai.networks.nets import UNet


class I2IModel(L.LightningModule):
    """PyTorch Lightning module for image-to-image translation with L1 loss only."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        lr: float = 1e-3,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Generator - 2D UNet
        self.generator = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
            act="relu"
        )
        
        # Loss
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x):
        return self.generator(x)
    
    def training_step(self, batch, batch_idx):
        input_img, target_img = batch['img'], batch['ref']
        
        # Generate prediction
        pred_img = self.forward(input_img)
        
        # L1 loss
        loss = self.loss_fn(pred_img, target_img)
        
        # Logging
        self.log("train/loss", loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """GridSampler-based sliding window validation with patch aggregation."""
        
        # Get volumes (1,1,X,Y,Z) - already on GPU
        input_volume = batch['img']  
        target_volume = batch['ref'] 
        
        # Remove batch dimension for TorchIO processing: (1,1,X,Y,Z) -> (1,X,Y,Z)
        input_vol = input_volume.squeeze(0).cpu()  # Move to CPU for TorchIO
        target_vol = target_volume.squeeze(0).cpu()
        
        # Create TorchIO Subject (no affine needed)
        subject = tio.Subject(
            img=tio.ScalarImage(tensor=input_vol),
            ref=tio.ScalarImage(tensor=target_vol)
        )
        
        # GridSampler for sliding window inference
        patch_size = (160, 160, 1)  # 2D patches from 3D volume
        patch_overlap = (64, 64, 0)  # 50% overlap in XY, no overlap in Z
        
        grid_sampler = tio.inference.GridSampler(
            subject,
            patch_size,
            patch_overlap
        )
        
        # GridAggregator to collect results
        aggregator = tio.inference.GridAggregator(
            grid_sampler,
            overlap_mode='average'
        )

        patch_loader = torch.utils.data.DataLoader(
                grid_sampler, batch_size=16, num_workers=4)

        # Process each patch
        self.eval()
        with torch.no_grad():
            for patch in patch_loader:
                # Get input patch (1,H,W,1) and squeeze to (1,H,W)
                patch_input = patch['img'][tio.DATA].squeeze(-1)
                
                # Move to GPU for inference
                patch_input = patch_input.to(self.device)
                
                # Forward pass through 2D generator
                patch_pred = self.forward(patch_input)
                
                # Move back to CPU and add Z dimension: (1,H,W) -> (1,H,W,1)
                patch_pred = patch_pred.cpu().unsqueeze(-1)
                locs = patch[tio.LOCATION] # Get Locations

                # Add to aggregator
                aggregator.add_batch(patch_pred, locs)
        
        # Get aggregated result
        pred_volume = aggregator.get_output_tensor()        
        # Add batch dimension back and move to GPU: (1,X,Y,Z) -> (1,1,X,Y,Z)
        pred_volume = pred_volume.unsqueeze(0).to(self.device)
        
        # Calculate validation loss
        val_loss = self.loss_fn(pred_volume, target_volume)
        
        # Logging
        self.log("val/loss", val_loss, prog_bar=True)
        
        return val_loss 
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.generator.parameters(), 
            lr=self.hparams.lr
        )