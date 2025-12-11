"""
Main training script for DETR moved object detection.
"""

import argparse
import torch
from torch.utils.data import DataLoader
import json
import os

from dataloader import MovedObjectDataset, collate_fn
from models import DETRPixelDiff, DETRFeatureDiff
from training import Trainer, get_trainable_parameters, FineTuneStrategy
from utils import set_seed, get_device, split_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DETR for moved object detection')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='.', help='Root data directory')
    parser.add_argument('--index_file', type=str, default='index.txt', help='Index file path')
    parser.add_argument('--matched_annotations_dir', type=str, default='data/matched_annotations',
                       help='Directory with matched annotations')
    parser.add_argument('--train_index', type=str, default=None, help='Train index file (if split already done)')
    parser.add_argument('--val_index', type=str, default=None, help='Val index file (if split already done)')
    
    # Model arguments
    parser.add_argument('--architecture', type=int, choices=[1, 2], default=2,
                       help='Architecture option: 1=feature diff, 2=pixel diff')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of object classes')
    parser.add_argument('--pretrained_model', type=str, default='facebook/detr-resnet-50',
                       help='Pretrained DETR model')
    parser.add_argument('--diff_amplify', type=float, default=1.0,
                       help='Multiplier for pixel difference (default: 1.0, try 5.0 to amplify signal)')
    parser.add_argument('--eos_coef', type=float, default=0.01,
                       help='Background class weight in loss (default: 0.01, lower = less background bias)')
    
    # Training arguments
    parser.add_argument('--finetune_strategy', type=str, default='full',
                       choices=['full', 'conv_only', 'classification_head_only', 'transformer_only'],
                       help='Fine-tuning strategy')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (recommended: 1e-4 to 5e-5 for DETR)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--image_size', type=int, default=800, help='Input image size')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='Max gradient norm for clipping')
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine'], help='LR scheduler type')
    
    # Other arguments
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--config', type=str, default=None, help='Config file path (JSON)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_model(architecture, num_classes, pretrained_model, **kwargs):
    """Create model based on architecture option."""
    if architecture == 1:
        # Feature difference architecture
        model = DETRFeatureDiff(
            num_classes=num_classes,
            pretrained_model=pretrained_model,
            use_detr_backbone=kwargs.get('use_detr_backbone', False),
            layer=kwargs.get('layer', 'layer3')
        )
    elif architecture == 2:
        # Pixel difference architecture
        model = DETRPixelDiff(
            num_classes=num_classes,
            pretrained_model=pretrained_model,
            diff_mode=kwargs.get('diff_mode', 'subtract')
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    else:
        config = vars(args)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Split data if needed
    if config.get('train_index') is None or config.get('val_index') is None:
        print("Splitting data into train/val sets...")
        train_index, val_index = split_data(
            config['index_file'],
            train_ratio=0.8,
            output_dir=config['output_dir']
        )
        config['train_index'] = train_index
        config['val_index'] = val_index
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MovedObjectDataset(
        index_file=config['train_index'],
        matched_annotations_dir=config['matched_annotations_dir'],
        data_dir=config['data_dir'],
        image_size=config['image_size']
    )
    
    val_dataset = MovedObjectDataset(
        index_file=config['val_index'],
        matched_annotations_dir=config['matched_annotations_dir'],
        data_dir=config['data_dir'],
        image_size=config['image_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print(f"Creating model (architecture {config['architecture']})...")
    model = create_model(
        architecture=config['architecture'],
        num_classes=config['num_classes'],
        pretrained_model=config['pretrained_model'],
        diff_amplify=config.get('diff_amplify', 1.0),
        eos_coef=config.get('eos_coef', 0.01)
    )
    
    # Apply fine-tuning strategy
    strategy_map = {
        'full': FineTuneStrategy.FULL,
        'conv_only': FineTuneStrategy.CONV_ONLY,
        'classification_head_only': FineTuneStrategy.CLASSIFICATION_HEAD_ONLY,
        'transformer_only': FineTuneStrategy.TRANSFORMER_ONLY
    }
    
    strategy = strategy_map[config['finetune_strategy']]
    print(f"Applying fine-tuning strategy: {strategy.value}")
    trainable_params = get_trainable_parameters(
        model, strategy, architecture_option=config['architecture']
    )
    
    # Create trainer
    trainer_config = {
        'learning_rate': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        'lr_step_size': config.get('lr_step_size', 30),
        'lr_gamma': config.get('lr_gamma', 0.1),
        'lr_scheduler': config.get('lr_scheduler', 'step'),
        'num_epochs': config['num_epochs'],
        'max_grad_norm': config.get('max_grad_norm', 0.1),
        'output_dir': config['output_dir'],
        'save_every': config['save_every']
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=config['num_epochs'])
    
    print(f"\nTraining completed! Results saved to {config['output_dir']}")


if __name__ == '__main__':
    main()

