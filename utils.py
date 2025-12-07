"""
Utility functions for the project.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_data(index_file, train_ratio=0.8, output_dir='.'):
    """
    Split data into train and test sets.
    
    Args:
        index_file: Path to index.txt file
        train_ratio: Ratio of data for training (default: 0.8)
        output_dir: Directory to save split files
    """
    # Read all pairs
    pairs = []
    with open(index_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                pairs.append(line.strip())
    
    # Shuffle
    random.shuffle(pairs)
    
    # Split
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    # Write split files
    train_file = os.path.join(output_dir, 'train_index.txt')
    test_file = os.path.join(output_dir, 'test_index.txt')
    
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_pairs))
    
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_pairs))
    
    print(f"Split {len(pairs)} pairs into {len(train_pairs)} train and {len(test_pairs)} test")
    return train_file, test_file


def get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

