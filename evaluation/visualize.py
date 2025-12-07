"""
Visualization utilities for moved object detection.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


# Color palette for different classes
CLASS_COLORS = {
    0: (128, 128, 128),  # Unknown - gray
    1: (255, 0, 0),      # Person - red
    2: (0, 255, 0),      # Car - green
    3: (0, 0, 255),      # Other vehicle - blue
    4: (255, 255, 0),    # Other object - cyan
    5: (255, 0, 255),    # Bike - magenta
}

CLASS_NAMES = {
    0: 'Unknown',
    1: 'Person',
    2: 'Car',
    3: 'Other Vehicle',
    4: 'Other Object',
    5: 'Bike'
}


def denormalize_bbox(bbox, img_width, img_height):
    """
    Convert normalized bbox [cx, cy, w, h] to pixel coordinates [x, y, w, h].
    
    Args:
        bbox: Normalized bbox [cx, cy, w, h]
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Pixel coordinates [x, y, w, h]
    """
    cx, cy, w, h = bbox
    x = (cx - w / 2) * img_width
    y = (cy - h / 2) * img_height
    w_px = w * img_width
    h_px = h * img_height
    return int(x), int(y), int(w_px), int(h_px)


def visualize_predictions(image1, image2, pred_boxes, pred_labels, pred_scores,
                         target_boxes=None, target_labels=None,
                         save_path=None, show=True):
    """
    Visualize predictions on images.
    
    Args:
        image1: First frame (numpy array or path)
        image2: Second frame (numpy array or path)
        pred_boxes: Predicted boxes [N, 4] normalized [cx, cy, w, h]
        pred_labels: Predicted labels [N]
        pred_scores: Prediction scores [N]
        target_boxes: Target boxes [M, 4] (optional)
        target_labels: Target labels [M] (optional)
        save_path: Path to save visualization (optional)
        show: Whether to display the image
    """
    # Load images if paths provided
    if isinstance(image1, (str, Path)):
        image1 = cv2.imread(str(image1))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    if isinstance(image2, (str, Path)):
        image2 = cv2.imread(str(image2))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # Convert to numpy if tensors
    if isinstance(image1, torch.Tensor):
        image1 = image1.permute(1, 2, 0).cpu().numpy()
        image1 = (image1 * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        image1 = np.clip(image1, 0, 1)
        image1 = (image1 * 255).astype(np.uint8)
    
    if isinstance(image2, torch.Tensor):
        image2 = image2.permute(1, 2, 0).cpu().numpy()
        image2 = (image2 * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        image2 = np.clip(image2, 0, 1)
        image2 = (image2 * 255).astype(np.uint8)
    
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.cpu().numpy()
    
    img_height, img_width = image2.shape[:2]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot frame 1
    axes[0].imshow(image1)
    axes[0].set_title('Frame 1 (Old)', fontsize=14)
    axes[0].axis('off')
    
    # Plot frame 2 with predictions
    axes[1].imshow(image2)
    axes[1].set_title('Frame 2 (New) - Predictions', fontsize=14)
    axes[1].axis('off')
    
    # Draw predictions on frame 2
    for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        if score < 0.5:  # Filter low confidence predictions
            continue
        
        x, y, w, h = denormalize_bbox(box, img_width, img_height)
        color = CLASS_COLORS.get(int(label), (255, 255, 255))
        color_normalized = tuple(c / 255.0 for c in color)
        
        # Draw bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                edgecolor=color_normalized, facecolor='none')
        axes[1].add_patch(rect)
        
        # Draw label
        label_text = f"{CLASS_NAMES.get(int(label), 'Unknown')} {score:.2f}"
        axes[1].text(x, y - 5, label_text, color=color_normalized, 
                    fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Draw ground truth if provided
    if target_boxes is not None and target_labels is not None:
        if isinstance(target_boxes, torch.Tensor):
            target_boxes = target_boxes.cpu().numpy()
        if isinstance(target_labels, torch.Tensor):
            target_labels = target_labels.cpu().numpy()
        
        for box, label in zip(target_boxes, target_labels):
            x, y, w, h = denormalize_bbox(box, img_width, img_height)
            color = CLASS_COLORS.get(int(label), (255, 255, 255))
            color_normalized = tuple(c / 255.0 for c in color)
            
            # Draw ground truth box with dashed line
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor=color_normalized, 
                                    facecolor='none', linestyle='--')
            axes[1].add_patch(rect)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_confusion_matrix(pred_labels, target_labels, num_classes=6, save_path=None):
    """
    Create and visualize confusion matrix.
    
    Args:
        pred_labels: List of predicted labels
        target_labels: List of target labels
        num_classes: Number of classes
        save_path: Path to save confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Flatten lists if nested
    if isinstance(pred_labels[0], (list, np.ndarray, torch.Tensor)):
        pred_labels = [item for sublist in pred_labels for item in sublist]
    if isinstance(target_labels[0], (list, np.ndarray, torch.Tensor)):
        target_labels = [item for sublist in target_labels for item in sublist]
    
    # Convert to numpy
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    if isinstance(target_labels, torch.Tensor):
        target_labels = target_labels.cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(target_labels, pred_labels, labels=range(num_classes))
    
    # Normalize
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=[CLASS_NAMES[i] for i in range(num_classes)],
                yticklabels=[CLASS_NAMES[i] for i in range(num_classes)])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=[CLASS_NAMES[i] for i in range(num_classes)],
                yticklabels=[CLASS_NAMES[i] for i in range(num_classes)])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

