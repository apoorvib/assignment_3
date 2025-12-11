"""
DETR with Feature Difference Architecture (Option 1).

This implementation extracts features from both images using ResNet,
computes the difference, and feeds it to the transformer block only.
"""

import torch
import torch.nn as nn
import math
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
        
        # Get transformer encoder and decoder from DETR
        # HuggingFace DETR has encoder and decoder separately
        self.encoder = self.detr.model.encoder
        self.decoder = self.detr.model.decoder
        
        # Position embeddings are handled internally by the encoder in HuggingFace DETR
        # We'll create sinusoidal position embeddings ourselves when needed
        self.position_embeddings = None
        
        # Get classification and bbox heads
        self.class_labels_classifier = self.detr.class_labels_classifier
        self.bbox_predictor = self.detr.bbox_predictor
        
        # Get object queries and query position embeddings
        # In HuggingFace DETR, these are typically accessed from the model
        # Try different possible locations
        self.query_position_embeddings = None
        self.object_queries = None
        
        # Try to find query position embeddings
        if hasattr(self.detr.model, 'query_position_embeddings'):
            self.query_position_embeddings = self.detr.model.query_position_embeddings
        elif hasattr(self.decoder, 'query_position_embeddings'):
            self.query_position_embeddings = self.decoder.query_position_embeddings
        elif hasattr(self.decoder, 'embed_positions'):
            self.query_position_embeddings = self.decoder.embed_positions
        
        # Try to find object queries (learnable query embeddings)
        if hasattr(self.detr.model, 'query_embeddings'):
            self.object_queries = self.detr.model.query_embeddings
        elif hasattr(self.decoder, 'query_embeddings'):
            self.object_queries = self.decoder.query_embeddings
        
        # If we can't find them, we'll need to create them or use decoder's default
        if self.query_position_embeddings is None or self.object_queries is None:
            # Get config to determine number of queries
            config = self.detr.config
            num_queries = getattr(config, 'num_queries', 100)
            hidden_dim = getattr(config, 'd_model', 256)
            
            # Create learnable object queries if not found
            if self.object_queries is None:
                self.object_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
            
            # Create learnable query position embeddings if not found
            if self.query_position_embeddings is None:
                self.query_position_embeddings = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
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
    
    def _create_sine_position_embeddings(self, num_positions, hidden_dim):
        """
        Create sinusoidal position embeddings similar to DETR.
        
        Args:
            num_positions: Number of positions (height * width)
            hidden_dim: Hidden dimension (should be 256 for DETR)
            
        Returns:
            Position embeddings tensor [num_positions, hidden_dim]
        """
        position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * 
                            (-math.log(10000.0) / hidden_dim))
        
        pos_embedding = torch.zeros(num_positions, hidden_dim)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_embedding
    
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
        
        # Create sinusoidal position embeddings
        seq_len = feat_diff_flat.shape[1]
        hidden_dim = feat_diff_flat.shape[2]
        pos_emb = self._create_sine_position_embeddings(seq_len, hidden_dim)
        position_embeddings = pos_emb.unsqueeze(0).to(feat_diff_flat.device)
        position_embeddings = position_embeddings.expand(batch_size, -1, -1)
        
        # Pass through transformer encoder
        # DetrEncoder expects inputs_embeds and object_queries (for position embeddings)
        encoder_outputs = self.encoder(
            inputs_embeds=feat_diff_flat,
            object_queries=position_embeddings,
                attention_mask=pixel_mask,
                output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Prepare object queries and query position embeddings
        # Handle different types of embeddings (Parameter, embedding layer, etc.)
        if isinstance(self.object_queries, nn.Parameter):
            object_queries = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1)
        elif hasattr(self.object_queries, 'weight'):
            object_queries = self.object_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            object_queries = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        if isinstance(self.query_position_embeddings, nn.Parameter):
            query_position_embeddings = self.query_position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        elif hasattr(self.query_position_embeddings, 'weight'):
            query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            query_position_embeddings = self.query_position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Pass through transformer decoder
        # Note: object_queries parameter in decoder is for encoder position embeddings
        decoder_outputs = self.decoder(
            inputs_embeds=object_queries,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=pixel_mask,
            object_queries=position_embeddings,  # Position embeddings for encoder outputs
            query_position_embeddings=query_position_embeddings
        )
        
        # Get decoder hidden states (these are the object queries after processing)
        object_queries = decoder_outputs.last_hidden_state
        
        # Classification and bbox prediction
        logits = self.class_labels_classifier(object_queries)
        pred_boxes = self.bbox_predictor(object_queries).sigmoid()
        
        # Create output similar to DETR's output format
        class DetrOutput:
            def __init__(self, logits, pred_boxes):
                self.logits = logits
                self.pred_boxes = pred_boxes
        
        return DetrOutput(logits=logits, pred_boxes=pred_boxes)

