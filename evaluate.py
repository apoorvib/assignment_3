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
    parser.add_argument('--diff_amplify', type=float, default=1.0,
                       help='Multiplier for pixel difference (must match training, default: 1.0)')
    parser.add_argument('--eos_coef', type=float, default=0.01,
                       help='Background class weight (must match training, default: 0.01)')
    
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
            diff_mode=kwargs.get('diff_mode', 'subtract'),
            diff_amplify=kwargs.get('diff_amplify', 1.0),
            eos_coef=kwargs.get('eos_coef', 0.01)
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
    # Background class is at index num_classes
    # We compare object class probabilities against background
    object_probs = probs[:, :, :num_classes]  # [batch, num_queries, num_classes]
    background_prob = probs[:, :, num_classes:num_classes+1]  # [batch, num_queries, 1]
    
    # Get max object class probability and label
    max_object_probs, labels = torch.max(object_probs, dim=-1)  # [batch, num_queries]
    
    # Score: probability that it's an object (not background)
    # Use the object probability directly, or compare against background
    scores = max_object_probs  # Simple: use max object class probability
    
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
        pretrained_model=args.pretrained_model,
        diff_amplify=args.diff_amplify,
        eos_coef=args.eos_coef
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
    total_predictions_before_threshold = 0
    total_predictions_after_threshold = 0
    max_scores_seen = []
    
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
            
            # Diagnostic: Check raw predictions before thresholding
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                # Get max scores (excluding background)
                max_scores = torch.max(probs[:, :, :args.num_classes], dim=-1)[0]
                max_scores_seen.extend(max_scores.cpu().numpy().flatten().tolist())
                total_predictions_before_threshold += (max_scores > 0.1).sum().item()  # Count predictions > 10%
            
            # Postprocess
            pred_boxes, pred_labels, pred_scores = postprocess_predictions(
                outputs, args.score_threshold, args.num_classes
            )
            
            # Count predictions after threshold
            for pred in pred_boxes:
                total_predictions_after_threshold += len(pred)
            
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
    
    # Print diagnostics
    print("\n" + "="*60)
    print("DIAGNOSTICS")
    print("="*60)
    if max_scores_seen:
        print(f"Max prediction scores seen: min={min(max_scores_seen):.4f}, max={max(max_scores_seen):.4f}, mean={sum(max_scores_seen)/len(max_scores_seen):.4f}")
        print(f"Predictions with score > 0.1: {total_predictions_before_threshold}")
        print(f"Predictions with score > {args.score_threshold}: {total_predictions_after_threshold}")
        print(f"Score threshold used: {args.score_threshold}")
        if total_predictions_after_threshold == 0 and total_predictions_before_threshold > 0:
            print(f"\n⚠️  WARNING: Model is making predictions but all scores are below threshold {args.score_threshold}")
            print(f"   Try lowering --score_threshold (e.g., --score_threshold 0.1)")
    else:
        print("No predictions detected - model may not be working correctly")
    print("="*60)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(
        all_pred_boxes, all_pred_labels, all_pred_scores,
        all_target_boxes, all_target_labels,
        iou_threshold=args.iou_threshold
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Architecture: Option {args.architecture} ({'Feature Diff' if args.architecture == 1 else 'Pixel Diff'})")
    print(f"IoU Threshold: {args.iou_threshold}")
    print(f"Score Threshold: {args.score_threshold}")
    print(f"Test Samples: {len(test_dataset)}")
    print("\n" + "-"*60)
    print("OVERALL METRICS")
    print("-"*60)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\n" + "-"*60)
    print("PER-CLASS METRICS")
    print("-"*60)
    for label in sorted(metrics['per_class_precision_recall'].keys()):
        pr = metrics['per_class_precision_recall'][label]
        ap = metrics['per_class_AP'].get(label, 0.0)
        class_names = {0: 'Unknown', 1: 'Person', 2: 'Car', 3: 'Other Vehicle', 4: 'Other Object', 5: 'Bike'}
        class_name = class_names.get(label, f'Class {label}')
        print(f"  {class_name} (Class {label}):")
        print(f"    Precision: {pr['precision']:.4f}, Recall: {pr['recall']:.4f}, AP: {ap:.4f}")
        print(f"    TP: {pr['tp']}, FP: {pr['fp']}, FN: {pr['fn']}")
    print("="*60)
    
    # Save results
    results_file = os.path.join(args.output_dir, 'results.json')
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k) if isinstance(k, (np.integer, np.int64, np.int32)) else k: convert_to_native(v) 
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    metrics_serializable = convert_to_native(metrics)
    
    with open(results_file, 'w') as f:
        json.dump(metrics_serializable, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    print(f"Visualizations saved to {os.path.join(args.output_dir, 'visualizations')}")


if __name__ == '__main__':
    main()

