"""
DataLoader for Moved Object Detection Dataset

This module provides a PyTorch Dataset class for loading image pairs
and their matched annotations (objects that moved between frames).
"""

import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class MovedObjectDataset(Dataset):
    """
    Dataset for moved object detection.
    
    Each sample contains:
    - Two images (frame1, frame2)
    - Matched annotations: objects that moved between frames
    - Each object has old bbox (in frame1) and new bbox (in frame2)
    """
    
    def __init__(self, index_file, matched_annotations_dir, data_dir='.', 
                 image_size=800, transform=None, split='train'):
        """
        Args:
            index_file: Path to index.txt file with image pairs
            matched_annotations_dir: Directory containing matched annotation files
            data_dir: Root directory for data (default: current directory)
            image_size: Target size for resizing images (default: 800)
            transform: Optional transform to apply to images
            split: 'train' or 'test' - used for filtering if needed
        """
        self.data_dir = data_dir
        self.matched_annotations_dir = matched_annotations_dir
        self.image_size = image_size
        
        # Load index file
        self.image_pairs = []
        with open(index_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    self.image_pairs.append(parts)
        
        # Filter matched annotation files that exist
        self.valid_pairs = []
        for pair in self.image_pairs:
            img1_path = os.path.join(data_dir, pair[0])
            img2_path = os.path.join(data_dir, pair[2])
            
            # Generate expected matched annotation filename
            folder_name = os.path.basename(os.path.dirname(img1_path))
            img1_name = os.path.splitext(os.path.basename(img1_path))[0]
            img2_name = os.path.splitext(os.path.basename(img2_path))[0]
            match_file = f"{folder_name}-{img1_name}-{img2_name}_match.txt"
            match_path = os.path.join(matched_annotations_dir, match_file)
            
            if os.path.exists(match_path) and os.path.exists(img1_path) and os.path.exists(img2_path):
                self.valid_pairs.append({
                    'img1': pair[0],
                    'ann1': pair[1],
                    'img2': pair[2],
                    'ann2': pair[3],
                    'match_file': match_file
                })
        
        print(f"Found {len(self.valid_pairs)} valid image pairs with matched annotations")
        
        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def load_image(self, img_path):
        """Load and preprocess an image."""
        full_path = os.path.join(self.data_dir, img_path)
        img = cv2.imread(full_path)
        if img is None:
            raise ValueError(f"Could not load image: {full_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize maintaining aspect ratio
        h, w = img.shape[:2]
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        pad_h = (self.image_size - new_h) // 2
        pad_w = (self.image_size - new_w) // 2
        img = cv2.copyMakeBorder(img, pad_h, self.image_size - new_h - pad_h,
                                pad_w, self.image_size - new_w - pad_w,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Convert to PIL Image for transforms
        img = Image.fromarray(img)
        
        return img, scale, (pad_w, pad_h)
    
    def parse_matched_annotations(self, match_file):
        """
        Parse matched annotation file.
        
        Format: Each object has 2 rows
        Row 1 (old): match_id x_old y_old w_old h_old type
        Row 2 (new): match_id x_new y_new w_new h_new type
        """
        match_path = os.path.join(self.matched_annotations_dir, match_file)
        objects = []
        
        with open(match_path, 'r') as f:
            lines = f.readlines()
            
        # Process pairs of lines (old, new)
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break
                
            # Parse old bbox
            old_parts = lines[i].strip().split()
            if len(old_parts) < 6:
                continue
            match_id = int(old_parts[0])
            x_old, y_old, w_old, h_old = map(float, old_parts[1:5])
            type_old = int(old_parts[5])
            
            # Parse new bbox
            new_parts = lines[i+1].strip().split()
            if len(new_parts) < 6:
                continue
            match_id_new = int(new_parts[0])
            x_new, y_new, w_new, h_new = map(float, new_parts[1:5])
            type_new = int(new_parts[5])
            
            # Verify match_id consistency
            if match_id != match_id_new:
                continue
            
            # We use the new bbox as the target (what we want to predict)
            # The old bbox is context information
            objects.append({
                'match_id': match_id,
                'old_bbox': [x_old, y_old, w_old, h_old],
                'new_bbox': [x_new, y_new, w_new, h_new],
                'object_type': type_new  # Use type from new frame
            })
        
        return objects
    
    def normalize_bbox(self, bbox, scale, pad, img_size):
        """
        Normalize bbox coordinates to [0, 1] range.
        
        Args:
            bbox: [x, y, w, h] in original image coordinates
            scale: Scale factor applied during resize
            pad: (pad_w, pad_h) padding applied
            img_size: Target image size
        """
        x, y, w, h = bbox
        
        # Apply scale
        x = x * scale
        y = y * scale
        w = w * scale
        h = h * scale
        
        # Apply padding
        x = x + pad[0]
        y = y + pad[1]
        
        # Convert to center format [cx, cy, w, h] and normalize
        cx = (x + w / 2) / img_size
        cy = (y + h / 2) / img_size
        w_norm = w / img_size
        h_norm = h / img_size
        
        return [cx, cy, w_norm, h_norm]
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        pair = self.valid_pairs[idx]
        
        # Load images
        img1, scale1, pad1 = self.load_image(pair['img1'])
        img2, scale2, pad2 = self.load_image(pair['img2'])
        
        # Apply transforms
        img1_tensor = self.transform(img1)
        img2_tensor = self.transform(img2)
        
        # Parse matched annotations
        objects = self.parse_matched_annotations(pair['match_file'])
        
        # Prepare targets for DETR
        # DETR expects: boxes (normalized cx, cy, w, h), labels
        boxes = []
        labels = []
        
        for obj in objects:
            # Normalize new bbox (this is what we want to predict)
            bbox_norm = self.normalize_bbox(obj['new_bbox'], scale2, pad2, self.image_size)
            boxes.append(bbox_norm)
            labels.append(obj['object_type'])
        
        # Convert to tensors
        if len(boxes) == 0:
            # No objects - return empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'image1': img1_tensor,
            'image2': img2_tensor,
            'boxes': boxes,
            'labels': labels,
            'image_id': idx,
            'pair_info': pair
        }


def collate_fn(batch):
    """
    Custom collate function for batching.
    DETR can handle variable number of objects per image.
    """
    images1 = torch.stack([item['image1'] for item in batch])
    images2 = torch.stack([item['image2'] for item in batch])
    
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    pair_info = [item['pair_info'] for item in batch]
    
    return {
        'images1': images1,
        'images2': images2,
        'boxes': boxes,
        'labels': labels,
        'image_ids': image_ids,
        'pair_info': pair_info
    }

