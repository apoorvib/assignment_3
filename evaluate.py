"""
Evaluation script for DETR moved object detection.
"""

import argparse
import torch
from torch.utils.data import DataLoader
import json
import os
import numpy as np

from dataloader import MovedObjectDataset, collate_fn
from models import DETRPixelDiff, DETRFeatureDiff
from evaluation import compute_metrics, visualize_predictions
from utils import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DETR for moved object detection')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='.', help='Root data directory')
    parser.add_argument('--test_index', type=str, default='test_index.txt', help='Test index file')
    parser.add_argument('--matched_annotations_dir', type=str, default='data/matched_annotations',
                       help='Directory with matched annotations')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--architecture', type=int, choices=[1, 2], default=2,
                       help='Architecture option: 1=feature diff, 2=pixel diff')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of object classes')
    parser.add_argument('--pretrained_model', type=str, default='facebook/detr-resnet-50',
                       help='Pretrained DETR model')
    parser.add_argument('--image_size', type=int, default=800, help='Input image size')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for matching')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Score threshold for predictions')
    parser.add_argument('--num_visualizations', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='eval_outputs', help='Output directory')
    
    return parser.parse_args()


def create_model(architecture, num_classes, pretrained_model, **kwargs):
    """Create model based on architecture option."""
    if architecture == 1:
        model = DETRFeatureDiff(
            num_classes=num_classes,
            pretrained_model=pretrained_model,
            use_detr_backbone=kwargs.get('use_detr_backbone', False),
            layer=kwargs.get('layer', 'layer3')
        )
    elif architecture == 2:
        model = DETRPixelDiff(
            num_classes=num_classes,
            pretrained_model=pretrained_model,
            diff_mode=kwargs.get('diff_mode', 'subtract')
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def postprocess_predictions(outputs, score_threshold=0.5, num_classes=6):
    """
    Postprocess DETR outputs to get boxes, labels, and scores.
    
    Args:
        outputs: Model outputs
        score_threshold: Score threshold for filtering
        num_classes: Number of classes
        
    Returns:
        boxes, labels, scores
    """
    if hasattr(outputs, 'logits'):
        logits = outputs.logits  # [batch, num_queries, num_classes+1]
        pred_boxes = outputs.pred_boxes  # [batch, num_queries, 4]
    else:
        logits = outputs['logits']
        pred_boxes = outputs['pred_boxes']
    
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)  # [batch, num_queries, num_classes+1]
    
    # Get scores and labels (excluding background class)
    scores, labels = torch.max(probs[:, :, :num_classes], dim=-1)  # [batch, num_queries]
    
    # Filter by score threshold
    batch_boxes = []
    batch_labels = []
    batch_scores = []
    
    for b in range(logits.shape[0]):
        mask = scores[b] >= score_threshold
        batch_boxes.append(pred_boxes[b][mask])
        batch_labels.append(labels[b][mask])
        batch_scores.append(scores[b][mask])
    
    return batch_boxes, batch_labels, batch_scores


def main():
    args = parse_args()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # Create dataset
    print("Creating test dataset...")
    test_dataset = MovedObjectDataset(
        index_file=args.test_index,
        matched_annotations_dir=args.matched_annotations_dir,
        data_dir=args.data_dir,
        image_size=args.image_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print(f"Creating model (architecture {args.architecture})...")
    model = create_model(
        architecture=args.architecture,
        num_classes=args.num_classes,
        pretrained_model=args.pretrained_model
    )
    
    # Load weights
    print(f"Loading model from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Evaluate
    print("\nEvaluating...")
    all_pred_boxes = []
    all_pred_labels = []
    all_pred_scores = []
    all_target_boxes = []
    all_target_labels = []
    
    visualization_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images1 = batch['images1'].to(device)
            images2 = batch['images2'].to(device)
            targets = {
                'boxes': batch['boxes'],
                'labels': batch['labels']
            }
            
            # Forward pass
            outputs = model(images1, images2)
            
            # Postprocess
            pred_boxes, pred_labels, pred_scores = postprocess_predictions(
                outputs, args.score_threshold, args.num_classes
            )
            
            # Collect predictions and targets
            for i in range(len(pred_boxes)):
                all_pred_boxes.append(pred_boxes[i].cpu())
                all_pred_labels.append(pred_labels[i].cpu())
                all_pred_scores.append(pred_scores[i].cpu())
                all_target_boxes.append(targets['boxes'][i])
                all_target_labels.append(targets['labels'][i])
            
            # Visualize some samples
            if visualization_count < args.num_visualizations:
                for i in range(min(len(pred_boxes), args.num_visualizations - visualization_count)):
                    pair_info = batch['pair_info'][i]
                    img1_path = os.path.join(args.data_dir, pair_info['img1'])
                    img2_path = os.path.join(args.data_dir, pair_info['img2'])
                    
                    visualize_predictions(
                        img1_path, img2_path,
                        pred_boxes[i].cpu(),
                        pred_labels[i].cpu(),
                        pred_scores[i].cpu(),
                        targets['boxes'][i],
                        targets['labels'][i],
                        save_path=os.path.join(
                            args.output_dir, 'visualizations',
                            f'sample_{visualization_count}.png'
                        ),
                        show=False
                    )
                    visualization_count += 1
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(
        all_pred_boxes, all_pred_labels, all_pred_scores,
        all_target_boxes, all_target_labels,
        iou_threshold=args.iou_threshold
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nPer-class metrics:")
    for label, pr in metrics['per_class_precision_recall'].items():
        print(f"  Class {label}: Precision={pr['precision']:.4f}, Recall={pr['recall']:.4f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    print(f"Visualizations saved to {os.path.join(args.output_dir, 'visualizations')}")


if __name__ == '__main__':
    main()

