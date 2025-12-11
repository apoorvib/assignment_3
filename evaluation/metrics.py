"""
Evaluation metrics for moved object detection.

Implements precision, recall, mAP, and other metrics.
"""

import torch
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    
    Args:
        box1: [cx, cy, w, h] normalized
        box2: [cx, cy, w, h] normalized
        
    Returns:
        IoU value
    """
    # Convert to [x1, y1, x2, y2]
    def to_corners(box):
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2
    
    x1_1, y1_1, x2_1, y2_1 = to_corners(box1)
    x1_2, y1_2, x2_2, y2_2 = to_corners(box2)
    
    # Intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_predictions(pred_boxes, pred_labels, target_boxes, target_labels, iou_threshold=0.5):
    """
    Match predictions to targets using IoU.
    
    Args:
        pred_boxes: List of predicted boxes [N, 4]
        pred_labels: List of predicted labels [N]
        target_boxes: List of target boxes [M, 4]
        target_labels: List of target labels [M]
        iou_threshold: IoU threshold for matching
        
    Returns:
        matched_indices: List of (pred_idx, target_idx) pairs
        unmatched_preds: List of unmatched prediction indices
        unmatched_targets: List of unmatched target indices
    """
    if len(pred_boxes) == 0:
        return [], [], list(range(len(target_boxes)))
    
    if len(target_boxes) == 0:
        return [], list(range(len(pred_boxes))), []
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(target_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, target_box in enumerate(target_boxes):
            iou_matrix[i, j] = compute_iou(pred_box, target_box)
    
    # Greedy matching
    matched_indices = []
    matched_preds = set()
    matched_targets = set()
    
    # Sort by IoU (highest first)
    matches = []
    for i in range(len(pred_boxes)):
        for j in range(len(target_boxes)):
            if pred_labels[i] == target_labels[j]:  # Same class
                matches.append((i, j, iou_matrix[i, j]))
    
    matches.sort(key=lambda x: x[2], reverse=True)
    
    for pred_idx, target_idx, iou in matches:
        if pred_idx not in matched_preds and target_idx not in matched_targets:
            if iou >= iou_threshold:
                matched_indices.append((pred_idx, target_idx))
                matched_preds.add(pred_idx)
                matched_targets.add(target_idx)
    
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_targets = [i for i in range(len(target_boxes)) if i not in matched_targets]
    
    return matched_indices, unmatched_preds, unmatched_targets


def compute_precision_recall(pred_boxes_list, pred_labels_list, pred_scores_list,
                            target_boxes_list, target_labels_list, iou_threshold=0.5):
    """
    Compute precision and recall.
    
    Args:
        pred_boxes_list: List of predicted boxes for each image
        pred_labels_list: List of predicted labels for each image
        pred_scores_list: List of prediction scores for each image
        target_boxes_list: List of target boxes for each image
        target_labels_list: List of target labels for each image
        iou_threshold: IoU threshold for matching
        
    Returns:
        precision: Overall precision
        recall: Overall recall
        per_class_metrics: Dict of per-class precision/recall
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    
    for pred_boxes, pred_labels, pred_scores, target_boxes, target_labels in zip(
        pred_boxes_list, pred_labels_list, pred_scores_list,
        target_boxes_list, target_labels_list
    ):
        # Convert to numpy if needed
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()
        if isinstance(target_boxes, torch.Tensor):
            target_boxes = target_boxes.cpu().numpy()
        if isinstance(target_labels, torch.Tensor):
            target_labels = target_labels.cpu().numpy()
        
        # Match predictions
        matched, unmatched_preds, unmatched_targets = match_predictions(
            pred_boxes, pred_labels, target_boxes, target_labels, iou_threshold
        )
        
        # Count TP, FP, FN
        total_tp += len(matched)
        total_fp += len(unmatched_preds)
        total_fn += len(unmatched_targets)
        
        # Per-class counts
        for pred_idx, target_idx in matched:
            label = target_labels[target_idx]
            per_class_tp[label] += 1
        
        for pred_idx in unmatched_preds:
            label = pred_labels[pred_idx]
            per_class_fp[label] += 1
        
        for target_idx in unmatched_targets:
            label = target_labels[target_idx]
            per_class_fn[label] += 1
    
    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    per_class_metrics = {}
    for label in set(list(per_class_tp.keys()) + list(per_class_fp.keys()) + list(per_class_fn.keys())):
        # Convert numpy types to Python native types for JSON serialization
        label_int = int(label) if isinstance(label, (np.integer, np.int64, np.int32)) else label
        tp = int(per_class_tp[label]) if isinstance(per_class_tp[label], (np.integer, np.int64, np.int32)) else per_class_tp[label]
        fp = int(per_class_fp[label]) if isinstance(per_class_fp[label], (np.integer, np.int64, np.int32)) else per_class_fp[label]
        fn = int(per_class_fn[label]) if isinstance(per_class_fn[label], (np.integer, np.int64, np.int32)) else per_class_fn[label]
        
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        
        per_class_metrics[label_int] = {
            'precision': float(prec),
            'recall': float(rec),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    return precision, recall, per_class_metrics


def compute_map(pred_boxes_list, pred_labels_list, pred_scores_list,
               target_boxes_list, target_labels_list, iou_threshold=0.5):
    """
    Compute mean Average Precision (mAP).
    
    Args:
        pred_boxes_list: List of predicted boxes for each image
        pred_labels_list: List of predicted labels for each image
        pred_scores_list: List of prediction scores for each image
        target_boxes_list: List of target boxes for each image
        target_labels_list: List of target labels for each image
        iou_threshold: IoU threshold for matching
        
    Returns:
        mAP: Mean Average Precision
        ap_per_class: Dict of AP per class
    """
    # Group by class
    class_predictions = defaultdict(list)
    class_targets = defaultdict(list)
    
    for pred_boxes, pred_labels, pred_scores, target_boxes, target_labels in zip(
        pred_boxes_list, pred_labels_list, pred_scores_list,
        target_boxes_list, target_labels_list
    ):
        # Convert to numpy if needed
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()
        if isinstance(target_boxes, torch.Tensor):
            target_boxes = target_boxes.cpu().numpy()
        if isinstance(target_labels, torch.Tensor):
            target_labels = target_labels.cpu().numpy()
        
        # Group by class
        for label in np.unique(pred_labels):
            mask = pred_labels == label
            class_predictions[label].append({
                'boxes': pred_boxes[mask],
                'scores': pred_scores[mask]
            })
        
        for label in np.unique(target_labels):
            mask = target_labels == label
            class_targets[label].append({
                'boxes': target_boxes[mask]
            })
    
    # Compute AP for each class
    ap_per_class = {}
    for label in set(list(class_predictions.keys()) + list(class_targets.keys())):
        # Convert numpy types to Python native types for JSON serialization
        label_int = int(label) if isinstance(label, (np.integer, np.int64, np.int32)) else label
        # Collect all predictions and targets for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_target_boxes = []
        
        for pred_dict in class_predictions.get(label, []):
            all_pred_boxes.extend(pred_dict['boxes'])
            all_pred_scores.extend(pred_dict['scores'])
        
        for target_dict in class_targets.get(label, []):
            all_target_boxes.extend(target_dict['boxes'])
        
        if len(all_pred_boxes) == 0:
            ap_per_class[label_int] = 0.0
            continue
        
        # Sort by score
        sorted_indices = np.argsort(all_pred_scores)[::-1]
        all_pred_boxes = np.array(all_pred_boxes)[sorted_indices]
        all_pred_scores = np.array(all_pred_scores)[sorted_indices]
        
        # Compute precision-recall curve
        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))
        
        matched_targets = set()
        for i, pred_box in enumerate(all_pred_boxes):
            best_iou = 0.0
            best_target_idx = -1
            
            for j, target_box in enumerate(all_target_boxes):
                if j in matched_targets:
                    continue
                iou = compute_iou(pred_box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j
            
            if best_iou >= iou_threshold and best_target_idx != -1:
                tp[i] = 1
                matched_targets.add(best_target_idx)
            else:
                fp[i] = 1
        
        # Compute cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Compute precision and recall
        num_targets = len(all_target_boxes)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (num_targets + 1e-6)
        
        # Compute AP (area under PR curve)
        ap = 0.0
        for r in np.arange(0, 1.1, 0.1):
            mask = recall >= r
            if np.any(mask):
                ap += np.max(precision[mask]) / 11.0
        
        ap_per_class[label_int] = float(ap)
    
    # Compute mAP
    if len(ap_per_class) > 0:
        map_value = np.mean(list(ap_per_class.values()))
    else:
        map_value = 0.0
    
    return map_value, ap_per_class


def compute_metrics(pred_boxes_list, pred_labels_list, pred_scores_list,
                   target_boxes_list, target_labels_list, iou_threshold=0.5):
    """
    Compute all metrics.
    
    Returns:
        Dictionary with all metrics
    """
    precision, recall, per_class_pr = compute_precision_recall(
        pred_boxes_list, pred_labels_list, pred_scores_list,
        target_boxes_list, target_labels_list, iou_threshold
    )
    
    map_value, ap_per_class = compute_map(
        pred_boxes_list, pred_labels_list, pred_scores_list,
        target_boxes_list, target_labels_list, iou_threshold
    )
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'mAP': float(map_value),
        'f1_score': float(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0),
        'per_class_precision_recall': per_class_pr,
        'per_class_AP': ap_per_class
    }

