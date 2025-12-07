"""
Base DETR model wrapper.
"""

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrImageProcessor


class DETRBase(nn.Module):
    """
    Base wrapper for DETR model from HuggingFace Transformers.
    """
    
    def __init__(self, num_classes=6, pretrained_model='facebook/detr-resnet-50'):
        """
        Args:
            num_classes: Number of object classes (default: 6 for VIRAT dataset)
            pretrained_model: HuggingFace model identifier
        """
        super().__init__()
        
        # Load pretrained DETR model
        self.detr = DetrForObjectDetection.from_pretrained(
            pretrained_model,
            num_labels=num_classes + 1,  # +1 for background class
            ignore_mismatched_sizes=True
        )
        
        self.num_classes = num_classes
        self.processor = DetrImageProcessor.from_pretrained(pretrained_model)
    
    def forward(self, pixel_values, pixel_mask=None, labels=None):
        """
        Forward pass through DETR.
        
        Args:
            pixel_values: Preprocessed image tensor [batch, channels, height, width]
            pixel_mask: Optional attention mask
            labels: Optional target labels for loss computation (list of dicts)
            
        Returns:
            DETR outputs with logits and pred_boxes (and loss if labels provided)
        """
        if labels is not None:
            outputs = self.detr(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        else:
            outputs = self.detr(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs
    
    def get_backbone(self):
        """Get the ResNet backbone from DETR."""
        return self.detr.model.backbone
    
    def get_transformer(self):
        """Get the transformer from DETR."""
        return self.detr.model.transformer
    
    def get_classification_head(self):
        """Get the classification head from DETR."""
        return self.detr.class_labels_classifier
    
    def get_bbox_head(self):
        """Get the bounding box regression head from DETR."""
        return self.detr.bbox_predictor

