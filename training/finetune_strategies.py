"""
Fine-tuning strategies for DETR models.

Implements different strategies for fine-tuning:
1. Full fine-tuning (all parameters)
2. Conv-only (only convolutional block) - not for Option 1
3. Classification head only
4. Transformer block only
"""

from enum import Enum
import torch.nn as nn


class FineTuneStrategy(Enum):
    """Enumeration of fine-tuning strategies."""
    FULL = "full"
    CONV_ONLY = "conv_only"
    CLASSIFICATION_HEAD_ONLY = "classification_head_only"
    TRANSFORMER_ONLY = "transformer_only"


def get_trainable_parameters(model, strategy, architecture_option=2):
    """
    Get trainable parameters based on fine-tuning strategy.
    
    Args:
        model: The DETR model
        strategy: FineTuneStrategy enum value
        architecture_option: 1 for feature diff, 2 for pixel diff
        
    Returns:
        List of parameters to train
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    trainable_params = []
    
    if strategy == FineTuneStrategy.FULL:
        # Unfreeze all parameters
        # Note: For Option 1, we don't fine-tune conv params (as per assignment)
        if architecture_option == 1:
            # Option 1: Feature diff - don't fine-tune conv params
            # Unfreeze transformer and heads
            if hasattr(model, 'transformer'):
                for param in model.transformer.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
            
            if hasattr(model, 'class_labels_classifier'):
                for param in model.class_labels_classifier.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
            
            if hasattr(model, 'bbox_predictor'):
                for param in model.bbox_predictor.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
            
            # Also unfreeze feature projection if exists
            if hasattr(model, 'feature_proj'):
                for param in model.feature_proj.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
        else:
            # Option 2: Pixel diff - can fine-tune everything
            for param in model.parameters():
                param.requires_grad = True
                trainable_params.append(param)
    
    elif strategy == FineTuneStrategy.CONV_ONLY:
        # Only fine-tune convolutional block
        # Note: Not applicable for Option 1
        if architecture_option == 1:
            raise ValueError("Conv-only fine-tuning not applicable for Option 1 (feature diff)")
        
        # For Option 2, fine-tune the backbone
        if hasattr(model, 'detr'):
            if hasattr(model.detr, 'model'):
                if hasattr(model.detr.model, 'backbone'):
                    for param in model.detr.model.backbone.parameters():
                        param.requires_grad = True
                        trainable_params.append(param)
    
    elif strategy == FineTuneStrategy.CLASSIFICATION_HEAD_ONLY:
        # Only fine-tune classification head
        if hasattr(model, 'class_labels_classifier'):
            for param in model.class_labels_classifier.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        elif hasattr(model, 'detr'):
            if hasattr(model.detr, 'class_labels_classifier'):
                for param in model.detr.class_labels_classifier.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
    
    elif strategy == FineTuneStrategy.TRANSFORMER_ONLY:
        # Only fine-tune transformer block and prediction heads
        # (heads must be trainable for gradient flow)
        if hasattr(model, 'transformer'):
            for param in model.transformer.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        elif hasattr(model, 'detr'):
            if hasattr(model.detr.model, 'transformer'):
                for param in model.detr.model.transformer.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
        
        # Also fine-tune position embeddings
        if hasattr(model, 'position_embeddings'):
            for param in model.position_embeddings.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        elif hasattr(model, 'detr'):
            if hasattr(model.detr.model, 'position_embeddings'):
                for param in model.detr.model.position_embeddings.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
        
        # IMPORTANT: Also unfreeze prediction heads (required for gradient flow)
        if hasattr(model, 'class_labels_classifier'):
            for param in model.class_labels_classifier.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        elif hasattr(model, 'detr'):
            if hasattr(model.detr, 'class_labels_classifier'):
                for param in model.detr.class_labels_classifier.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
        
        if hasattr(model, 'bbox_predictor'):
            for param in model.bbox_predictor.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        elif hasattr(model, 'detr'):
            if hasattr(model.detr, 'bbox_predictor'):
                for param in model.detr.bbox_predictor.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Count trainable parameters
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    
    print(f"Strategy: {strategy.value}")
    print(f"Trainable parameters: {num_trainable:,} / {num_total:,} ({100*num_trainable/num_total:.2f}%)")
    
    return trainable_params


def freeze_all_except(model, module_names):
    """
    Freeze all parameters except those in specified modules.
    
    Args:
        model: The model
        module_names: List of module names to keep trainable
    """
    for name, param in model.named_parameters():
        is_trainable = any(name.startswith(prefix) for prefix in module_names)
        param.requires_grad = is_trainable

