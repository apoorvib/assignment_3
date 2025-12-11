"""
Trainer class for DETR fine-tuning.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from transformers import DetrImageProcessor


class Trainer:
    """
    Trainer for DETR models.
    """
    
    def __init__(self, model, train_loader, val_loader, config, device):
        """
        Args:
            model: DETR model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_step_size', 30),
            gamma=config.get('lr_gamma', 0.1)
        )
        
        # Loss function (DETR uses its own loss)
        # We'll use the loss from the model's forward pass
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_class_error': [],
            'val_class_error': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Output directory
        self.output_dir = config.get('output_dir', 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def format_targets_for_detr(self, targets):
        """
        Format targets for HuggingFace DETR.
        
        Args:
            targets: Dict with 'boxes' and 'labels' lists
            
        Returns:
            List of target dicts in DETR format
        """
        detr_targets = []
        for boxes, labels in zip(targets['boxes'], targets['labels']):
            # HuggingFace DETR expects:
            # - boxes: tensor [N, 4] with [x_center, y_center, width, height] normalized
            # - class_labels: tensor [N] with class indices
            target_dict = {
                'boxes': boxes.to(self.device),
                'class_labels': labels.to(self.device)
            }
            detr_targets.append(target_dict)
        return detr_targets
    
    def compute_loss(self, outputs, targets):
        """
        Compute DETR loss.
        
        Args:
            outputs: Model outputs (from forward pass)
            targets: Target dictionaries with 'boxes' and 'labels'
            
        Returns:
            Loss dictionary
        """
        # If outputs already have loss (from HuggingFace DETR), use it
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            return {
                'loss': outputs.loss,
                'loss_class_error': getattr(outputs, 'loss_dict', {}).get('loss_class_error', torch.tensor(0.0))
            }
        
        # Otherwise, compute loss manually (for Option 1 or if labels weren't passed)
        detr_targets = self.format_targets_for_detr(targets)
        loss_dict = self._compute_simple_loss(outputs, detr_targets)
        return loss_dict
    
    def _compute_simple_loss(self, outputs, targets):
        """
        Simplified loss computation.
        Note: DETR's actual loss uses bipartite matching.
        This is a placeholder - should use proper DETR loss.
        """
        # Get predictions
        if hasattr(outputs, 'logits'):
            logits = outputs.logits  # [batch, num_queries, num_classes+1]
            pred_boxes = outputs.pred_boxes  # [batch, num_queries, 4]
        else:
            logits = outputs['logits']
            pred_boxes = outputs['pred_boxes']
        
        # Classification loss (simplified - should use bipartite matching)
        batch_size = logits.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            target_boxes = targets[i]['boxes']
            target_labels = targets[i]['class_labels']
            
            if len(target_boxes) == 0:
                # No objects - penalize all predictions as background
                cls_loss = nn.functional.cross_entropy(
                    logits[i].view(-1, logits.shape[-1]),
                    torch.zeros(logits.shape[1], dtype=torch.long, device=self.device)
                )
                total_loss += cls_loss
            else:
                # Simple matching: use first num_objects queries
                num_objects = len(target_boxes)
                pred_logits = logits[i, :num_objects]
                pred_boxes_i = pred_boxes[i, :num_objects]
                
                # Classification loss
                cls_loss = nn.functional.cross_entropy(
                    pred_logits,
                    target_labels
                )
                
                # Bbox loss (L1 + GIoU)
                pred_boxes_flat = pred_boxes_i
                target_boxes_flat = target_boxes
                
                # L1 loss
                l1_loss = nn.functional.l1_loss(pred_boxes_flat, target_boxes_flat)
                
                # GIoU loss (simplified - should use proper GIoU)
                giou_loss = 1.0 - self._compute_giou(pred_boxes_flat, target_boxes_flat)
                
                total_loss += cls_loss + 5.0 * l1_loss + 2.0 * giou_loss
        
        return {
            'loss': total_loss / batch_size,
            'loss_class_error': torch.tensor(0.0),  # Placeholder
        }
    
    def _compute_giou(self, pred_boxes, target_boxes):
        """
        Compute Generalized IoU (simplified version).
        """
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        def to_corners(boxes):
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return torch.stack([x1, y1, x2, y2], dim=1)
        
        pred_corners = to_corners(pred_boxes)
        target_corners = to_corners(target_boxes)
        
        # Compute IoU
        inter_x1 = torch.max(pred_corners[:, 0], target_corners[:, 0])
        inter_y1 = torch.max(pred_corners[:, 1], target_corners[:, 1])
        inter_x2 = torch.min(pred_corners[:, 2], target_corners[:, 2])
        inter_y2 = torch.min(pred_corners[:, 3], target_corners[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        pred_area = (pred_corners[:, 2] - pred_corners[:, 0]) * (pred_corners[:, 3] - pred_corners[:, 1])
        target_area = (target_corners[:, 2] - target_corners[:, 0]) * (target_corners[:, 3] - target_corners[:, 1])
        
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        
        # Simplified GIoU (full implementation is more complex)
        return iou.mean()
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            images1 = batch['images1'].to(self.device)
            images2 = batch['images2'].to(self.device)
            targets = {
                'boxes': batch['boxes'],
                'labels': batch['labels']
            }
            
            # Forward pass with labels for loss computation
            self.optimizer.zero_grad()
            
            # Format targets for DETR
            detr_targets = self.format_targets_for_detr(targets)
            
            # For Option 2 (pixel diff), pass labels to get built-in loss
            # For Option 1 (feature diff), we'll compute loss manually
            if hasattr(self.model, 'detr') and hasattr(self.model, 'compute_diff'):
                # This is Option 2 - can use built-in loss by passing labels
                outputs = self.model(images1, images2, labels=detr_targets)
            else:
                # This is Option 1 - forward without labels, compute loss manually
                outputs = self.model(images1, images2)
            
            # Compute loss
            loss_dict = self.compute_loss(outputs, targets)
            loss = loss_dict['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        self.history['train_loss'].append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                images1 = batch['images1'].to(self.device)
                images2 = batch['images2'].to(self.device)
                targets = {
                    'boxes': batch['boxes'],
                    'labels': batch['labels']
                }
                
                # Forward pass (validation - no labels needed)
                outputs = self.model(images1, images2)
                
                # Compute loss
                loss_dict = self.compute_loss(outputs, targets)
                loss = loss_dict['loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        self.history['val_loss'].append(avg_loss)
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_model_state = self.model.state_dict().copy()
            torch.save(self.best_model_state, 
                      os.path.join(self.output_dir, 'best_model.pth'))
        
        return avg_loss
    
    def train(self, num_epochs):
        """Train the model for multiple epochs."""
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'history': self.history
                }
                torch.save(checkpoint, 
                          os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save final model and history
        torch.save(self.model.state_dict(), 
                  os.path.join(self.output_dir, 'final_model.pth'))
        
        with open(os.path.join(self.output_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total epochs: {num_epochs}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final train loss: {self.history['train_loss'][-1]:.4f}")
        print(f"Final val loss: {self.history['val_loss'][-1]:.4f}")
        print(f"\nBest model saved to: {os.path.join(self.output_dir, 'best_model.pth')}")
        print(f"Final model saved to: {os.path.join(self.output_dir, 'final_model.pth')}")
        print(f"Training history saved to: {os.path.join(self.output_dir, 'history.json')}")
        print("="*60)

