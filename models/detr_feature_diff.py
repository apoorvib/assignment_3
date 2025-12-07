"""
DETR with Feature Difference Architecture (Option 1).

This implementation extracts features from both images using ResNet,
computes the difference, and feeds it to the transformer block only.
"""

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrImageProcessor
import torchvision.models as models


class DETRFeatureDiff(nn.Module):
    """
    DETR model that uses feature difference approach.
    
    Architecture:
    1. Pass img1 and img2 separately through ResNet50 (up to layer 3)
    2. Compute feature difference: feat_diff = feat2 - feat1
    3. Feed feat_diff to DETR transformer block only
    """
    
    def __init__(self, num_classes=6, pretrained_model='facebook/detr-resnet-50',
                 use_detr_backbone=False, layer='layer3'):
        """
        Args:
            num_classes: Number of object classes
            pretrained_model: HuggingFace model identifier
            use_detr_backbone: If True, use forward hooks on DETR's ResNet
                              If False, use separate ImageNet ResNet50
            layer: Which layer to extract features from ('layer3' or 'layer4')
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_detr_backbone = use_detr_backbone
        self.layer = layer
        
        # Load full DETR model (we'll use parts of it)
        self.detr = DetrForObjectDetection.from_pretrained(
            pretrained_model,
            num_labels=num_classes + 1,
            ignore_mismatched_sizes=True
        )
        
        self.processor = DetrImageProcessor.from_pretrained(pretrained_model)
        
        if use_detr_backbone:
            # Use DETR's internal ResNet with forward hooks
            self.backbone = self.detr.model.backbone
            self._setup_hooks()
        else:
            # Use separate ImageNet ResNet50
            resnet = models.resnet50(pretrained=True)
            
            # Extract up to specified layer
            if layer == 'layer3':
                self.backbone = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                    resnet.layer2,
                    resnet.layer3
                )
                # Get feature dimensions from layer3
                self.feature_dim = 1024
            elif layer == 'layer4':
                self.backbone = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                    resnet.layer2,
                    resnet.layer3,
                    resnet.layer4
                )
                self.feature_dim = 2048
            else:
                raise ValueError(f"Unknown layer: {layer}")
        
        # Get transformer from DETR
        self.transformer = self.detr.model.transformer
        
        # Get position embeddings (needed for transformer)
        self.position_embeddings = self.detr.model.position_embeddings
        
        # Get classification and bbox heads
        self.class_labels_classifier = self.detr.class_labels_classifier
        self.bbox_predictor = self.detr.bbox_predictor
        
        # Get object queries
        self.query_position_embeddings = self.detr.model.query_position_embeddings
        
        # Projection layer to match DETR's expected input dimension
        # DETR expects 256-dim features, ResNet outputs 1024 or 2048
        if not use_detr_backbone:
            self.feature_proj = nn.Conv2d(
                self.feature_dim, 256,
                kernel_size=1
            )
        else:
            # DETR backbone already outputs correct dimension
            self.feature_proj = nn.Identity()
    
    def _setup_hooks(self):
        """Setup forward hooks to extract features from DETR's backbone."""
        self.features = {}
        
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        # Register hook on the backbone output
        # DETR backbone returns a dict with '0', '1', '2', '3' keys
        # We'll use the last one
        if hasattr(self.backbone, 'conv_encoder'):
            # This is the ResNet part
            if self.layer == 'layer3':
                # Hook into layer3
                self.backbone.conv_encoder.model.layer3.register_forward_hook(
                    get_features('layer3')
                )
            elif self.layer == 'layer4':
                self.backbone.conv_encoder.model.layer4.register_forward_hook(
                    get_features('layer4')
                )
    
    def extract_features(self, image):
        """
        Extract features from an image using ResNet.
        
        Args:
            image: Image tensor [batch, channels, height, width]
            
        Returns:
            Feature tensor [batch, channels, height, width]
        """
        if self.use_detr_backbone:
            # Use DETR's backbone with hooks
            _ = self.backbone(image)
            features = self.features[self.layer]
        else:
            # Use separate ResNet
            features = self.backbone(image)
        
        # Project to DETR's expected dimension
        features = self.feature_proj(features)
        return features
    
    def forward(self, image1, image2, pixel_mask=None, labels=None):
        """
        Forward pass.
        
        Args:
            image1: First frame [batch, channels, height, width]
            image2: Second frame [batch, channels, height, width]
            pixel_mask: Optional attention mask
            
        Returns:
            DETR-style outputs with logits and pred_boxes
        """
        # Extract features from both images
        feat1 = self.extract_features(image1)
        feat2 = self.extract_features(image2)
        
        # Compute feature difference
        feat_diff = feat2 - feat1
        
        # Prepare for transformer
        # DETR expects features in format [batch, channels, height, width]
        # and converts to [batch, sequence_length, hidden_size]
        batch_size, channels, height, width = feat_diff.shape
        
        # Flatten spatial dimensions
        feat_diff_flat = feat_diff.flatten(2).transpose(1, 2)  # [batch, h*w, channels]
        
        # Add position embeddings
        position_embeddings = self.position_embeddings(feat_diff_flat)
        feat_diff_flat = feat_diff_flat + position_embeddings
        
        # Prepare object queries
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0)
        query_position_embeddings = query_position_embeddings.expand(batch_size, -1, -1)
        
        # Pass through transformer
        # Transformer expects: inputs_embeds, attention_mask, query_position_embeddings
        transformer_outputs = self.transformer(
            inputs_embeds=feat_diff_flat,
            attention_mask=pixel_mask,
            query_position_embeddings=query_position_embeddings
        )
        
        # Get hidden states
        hidden_states = transformer_outputs.last_hidden_state
        
        # Split into object queries and image features
        # DETR uses first N queries for objects, rest for image features
        num_queries = self.query_position_embeddings.weight.shape[0]
        object_queries = hidden_states[:, :num_queries, :]
        
        # Classification and bbox prediction
        logits = self.class_labels_classifier(object_queries)
        pred_boxes = self.bbox_predictor(object_queries).sigmoid()
        
        # Create output similar to DETR's output format
        class DetrOutput:
            def __init__(self, logits, pred_boxes):
                self.logits = logits
                self.pred_boxes = pred_boxes
        
        return DetrOutput(logits=logits, pred_boxes=pred_boxes)

