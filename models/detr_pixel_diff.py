"""
DETR with Pixel Difference Architecture (Option 2).

This implementation computes pixel-wise difference between two frames
and feeds it to the entire DETR model.
"""

import torch
import torch.nn as nn
from .detr_base import DETRBase


class DETRPixelDiff(DETRBase):
    """
    DETR model that takes pixel difference of two images as input.
    
    Architecture:
    1. Compute pixel-wise difference: img_diff = img2 - img1
    2. Feed img_diff to entire DETR model
    """
    
    def __init__(self, num_classes=6, pretrained_model='facebook/detr-resnet-50', 
                 diff_mode='subtract'):
        """
        Args:
            num_classes: Number of object classes
            pretrained_model: HuggingFace model identifier
            diff_mode: How to compute difference
                - 'subtract': img2 - img1
                - 'abs': |img2 - img1|
                - 'concat': Concatenate along channel dimension
        """
        super().__init__(num_classes, pretrained_model)
        self.diff_mode = diff_mode
        
        # If using concat, need to adjust input channels
        if diff_mode == 'concat':
            # Replace first conv layer to accept 6 channels (3+3)
            try:
                old_conv = self.detr.model.backbone.conv_encoder.model.conv1
            except AttributeError as e:
                raise AttributeError(
                    f"Could not access conv1 layer. DETR backbone structure may have changed. "
                    f"Original error: {e}"
                )
            
            new_conv = nn.Conv2d(
                6, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # Initialize with pretrained weights (replicate the 3 channels for both input images)
            with torch.no_grad():
                # Copy weights: first 3 channels from original, last 3 channels also from original
                new_conv.weight.data[:, :3] = old_conv.weight.data
                new_conv.weight.data[:, 3:] = old_conv.weight.data
                new_conv.weight.data = new_conv.weight.data / 2
                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias.data.clone()
            self.detr.model.backbone.conv_encoder.model.conv1 = new_conv
    
    def compute_diff(self, img1, img2):
        """
        Compute difference between two images.
        
        Args:
            img1: First image tensor [batch, channels, height, width]
            img2: Second image tensor [batch, channels, height, width]
            
        Returns:
            Difference tensor
        """
        # Validate input shapes
        if img1.shape != img2.shape:
            raise ValueError(
                f"Image shapes must match for pixel difference computation. "
                f"Got img1: {img1.shape}, img2: {img2.shape}"
            )
        
        if self.diff_mode == 'subtract':
            return img2 - img1
        elif self.diff_mode == 'abs':
            return torch.abs(img2 - img1)
        elif self.diff_mode == 'concat':
            return torch.cat([img1, img2], dim=1)
        else:
            raise ValueError(f"Unknown diff_mode: {self.diff_mode}")
    
    def forward(self, image1, image2, pixel_mask=None, labels=None):
        """
        Forward pass.
        
        Args:
            image1: First frame [batch, channels, height, width]
            image2: Second frame [batch, channels, height, width]
            pixel_mask: Optional attention mask
            labels: Optional target labels for loss computation (list of dicts)
            
        Returns:
            DETR outputs (with loss if labels provided)
        """
        # Compute pixel difference
        img_diff = self.compute_diff(image1, image2)
        
        # Feed to DETR with labels if provided (for loss computation)
        if labels is not None:
            outputs = self.detr(pixel_values=img_diff, pixel_mask=pixel_mask, labels=labels)
        else:
            outputs = self.detr(pixel_values=img_diff, pixel_mask=pixel_mask)
        return outputs

