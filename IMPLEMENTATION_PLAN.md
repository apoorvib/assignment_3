# Implementation Plan for Assignment 3: Moved Object Detection Using DETR

## Overview
This document outlines the step-by-step plan to build an end-to-end DETR pipeline for detecting moved objects between two frames of surveillance footage.

---

## Phase 1: Data Preparation & Understanding (FIRST PRIORITY)

### 1.1 Fix/Verify Ground Truth Labeller
- **Issue**: The labeller's `run_pipeline` function references paths that may not exist (`../data/base/cv_data_hw2`)
- **Action**: 
  - Update paths to use current `data/` directory and `index.txt` in root
  - Verify the labeller generates matched annotations correctly
  - Test on a small subset first

### 1.2 Generate Matched Annotations
- **Action**: Run `data_ground_truth_labeller.py` to generate matched annotation files
- **Output Format**: Each matched object has 2 rows:
  - Row 1: `<match_id> <x_old> <y_old> <w_old> <h_old> <object_type>`
  - Row 2: `<match_id> <x_new> <y_new> <w_new> <h_new> <object_type>`
- **Location**: Store in `data/matched_annotations/` directory

### 1.3 Create Dataset Class
- **File**: `dataloader.py`
- **Components**:
  - `MovedObjectDataset` class that:
    - Loads image pairs (frame1, frame2)
    - Parses matched annotation files
    - Returns both old and new bounding boxes for each moved object
    - Handles image preprocessing (resize, normalize, etc.)
    - Handles different image sizes (as mentioned in README)
  - Transform pipeline for images
  - Collate function for batching

### 1.4 Data Splitting
- **Action**: Create train/test split (80-20)
- **File**: `utils.py` or `data_split.py`
- **Function**: Split matched annotation files into train/test sets
- **Output**: Two index files (`train_index.txt`, `test_index.txt`)

### 1.5 Understand DETR Input/Output Format
- **Research**: 
  - DETR expects COCO-style annotations (normalized bbox coordinates)
  - DETR outputs: `{'logits': ..., 'pred_boxes': ...}`
  - DETR loss uses bipartite matching
- **Action**: Document expected formats for our use case

---

## Phase 2: Model Architecture Implementation (SECOND PRIORITY)

### 2.1 Base DETR Setup
- **File**: `models/detr_base.py`
- **Action**: 
  - Load `facebook/detr-resnet-50` from HuggingFace Transformers
  - Understand model structure (backbone, transformer, heads)
  - Create base wrapper class

### 2.2 Implement Option 1: Feature Difference Architecture
- **File**: `models/detr_feature_diff.py`
- **Architecture**:
  1. Pass image1 and image2 separately through ResNet50 backbone
  2. Extract features (up to layer 3, before FC layers)
  3. Compute feature difference: `feat_diff = feat2 - feat1`
  4. Feed `feat_diff` to DETR transformer block only (NOT entire model)
- **Implementation Options**:
  - **Option A**: Use separate ImageNet ResNet50, extract up to layer 3
  - **Option B**: Use forward hooks on DETR's internal ResNet (more flexible)
- **Recommendation**: Start with Option A (simpler), then try Option B

### 2.3 Implement Option 2: Pixel Difference Architecture
- **File**: `models/detr_pixel_diff.py`
- **Architecture**:
  1. Compute pixel-wise difference: `img_diff = img2 - img1`
  2. Handle size mismatches (resize to same size if needed)
  3. Feed `img_diff` to entire DETR model
- **Considerations**:
  - Image normalization after difference
  - Handling negative values in difference
  - Potentially use absolute difference or other operations

### 2.4 Model Configuration
- **File**: `configs/model_config.py`
- **Parameters**:
  - Architecture choice (Option 1 or 2)
  - Number of object classes (6: Unknown, person, car, other vehicle, other object, bike)
  - Input image size
  - Model-specific hyperparameters

---

## Phase 3: Training Infrastructure (THIRD PRIORITY)

### 3.1 Loss Function
- **File**: `training/loss.py`
- **Action**: 
  - DETR uses custom loss with bipartite matching
  - HuggingFace Transformers provides this, but may need adaptation
  - Handle both old and new bbox predictions (we need to predict new positions)

### 3.2 Fine-tuning Strategies
- **File**: `training/finetune_strategies.py`
- **Strategies to implement**:
  1. **Full fine-tuning**: All parameters (except conv if Option 1)
  2. **Conv-only**: Only convolutional block (not for Option 1)
  3. **Classification head only**: Only transformer classification head
  4. **Transformer block only**: Entire transformer or subset of layers
- **Function**: `get_trainable_parameters(model, strategy)` that freezes/unfreezes layers

### 3.3 Trainer Class
- **File**: `training/trainer.py`
- **Components**:
  - Training loop
  - Validation loop
  - Checkpoint saving/loading
  - Learning rate scheduling
  - Early stopping (optional)
  - Logging (tensorboard or simple logging)

### 3.4 Configuration System
- **File**: `configs/train_config.py`
- **Parameters**:
  - Learning rate
  - Batch size
  - Number of epochs
  - Optimizer (AdamW recommended for DETR)
  - Fine-tuning strategy
  - Architecture option
  - Data paths
  - Output directories

### 3.5 Main Training Script
- **File**: `train.py`
- **Action**: 
  - Parse config
  - Initialize dataset, dataloader
  - Initialize model
  - Initialize trainer
  - Run training

---

## Phase 4: Evaluation & Metrics (FOURTH PRIORITY)

### 4.1 Evaluation Metrics
- **File**: `evaluation/metrics.py`
- **Metrics to implement**:
  - **Precision**: TP / (TP + FP)
  - **Recall**: TP / (TP + FN)
  - **mAP** (mean Average Precision): Standard object detection metric
  - **IoU-based matching**: For determining TP/FP/FN
- **Considerations**:
  - We predict new bbox positions for moved objects
  - Need to match predictions with ground truth (IoU threshold)

### 4.2 Evaluation Script
- **File**: `evaluate.py`
- **Components**:
  - Load trained model
  - Run on test set
  - Compute all metrics
  - Generate summary report

### 4.3 Visualization Tools
- **File**: `evaluation/visualize.py`
- **Visualizations**:
  - Predicted vs ground truth bboxes on images
  - Confusion matrix
  - Precision-Recall curves
  - Sample predictions (qualitative)
- **Output**: Save visualization images

---

## Phase 5: Experimentation & SLURM Setup (FIFTH PRIORITY)

### 5.1 SLURM Scripts
- **Files**: 
  - `scripts/train.slurm` - Training job
  - `scripts/eval.slurm` - Evaluation job
- **Components**:
  - Resource allocation (GPU, memory, time)
  - Environment setup (conda/venv)
  - Command execution
  - Output redirection

### 5.2 Experiment Configuration
- **File**: `configs/experiments/`
- **Create configs for each experiment**:
  - `exp1_option1_full_finetune.yaml`
  - `exp2_option1_classification_head.yaml`
  - `exp3_option1_transformer_only.yaml`
  - `exp4_option2_full_finetune.yaml`
  - `exp5_option2_conv_only.yaml`
  - `exp6_option2_classification_head.yaml`
  - `exp7_option2_transformer_only.yaml`
- **Note**: Some experiments may not apply (e.g., conv-only for Option 1)

### 5.3 Run Experiments
- **Action**: Execute all training runs
- **Documentation**: Keep track of:
  - Training losses
  - Validation metrics
  - Training time
  - Best checkpoints

### 5.4 Ablation Studies
- **File**: `ablation_studies.py` or separate configs
- **Studies**:
  - Different learning rates
  - Different batch sizes
  - Different transformer layer subsets
  - Different image preprocessing
  - Option 1 vs Option 2 comparison

---

## Phase 6: Documentation & Submission (FINAL PRIORITY)

### 6.1 README
- **File**: `README.md` (update existing)
- **Contents**:
  - Project overview
  - Installation instructions
  - Data preparation steps
  - How to run training
  - How to run evaluation
  - How to use SLURM scripts
  - Project structure

### 6.2 Code Organization
- **Directory Structure**:
```
assignment_3/
├── data/
│   ├── matched_annotations/
│   └── ...
├── models/
│   ├── detr_base.py
│   ├── detr_feature_diff.py
│   └── detr_pixel_diff.py
├── training/
│   ├── trainer.py
│   ├── loss.py
│   └── finetune_strategies.py
├── evaluation/
│   ├── metrics.py
│   └── visualize.py
├── configs/
│   ├── model_config.py
│   ├── train_config.py
│   └── experiments/
├── scripts/
│   ├── train.slurm
│   └── eval.slurm
├── utils.py
├── dataloader.py
├── train.py
├── evaluate.py
├── data_ground_truth_labeller.py
└── README.md
```

### 6.3 Technical Report Outline
- **Sections**:
  1. Introduction & Problem Statement
  2. Related Work (DETR, object detection)
  3. Methodology:
     - Architecture choices (Option 1 vs 2)
     - Fine-tuning strategies
     - Data preprocessing
  4. Experiments:
     - Setup (hardware, hyperparameters)
     - Results (tables, graphs)
     - Ablation studies
  5. Discussion:
     - Design choices justification
     - Limitations
     - Future work
  6. Conclusion

---

## Implementation Order Summary

### Week 1: Data & Model Foundation
1. ✅ Fix/verify ground truth labeller
2. ✅ Generate matched annotations
3. ✅ Create dataset class
4. ✅ Implement base DETR wrapper
5. ✅ Implement Option 1 architecture
6. ✅ Implement Option 2 architecture

### Week 2: Training & Evaluation
7. ✅ Create training infrastructure
8. ✅ Implement fine-tuning strategies
9. ✅ Create evaluation metrics
10. ✅ Create visualization tools
11. ✅ Test on small subset

### Week 3: Experiments & Documentation
12. ✅ Create SLURM scripts
13. ✅ Run all experiments
14. ✅ Collect results
15. ✅ Generate visualizations
16. ✅ Write technical report
17. ✅ Finalize README

---

## Key Design Decisions to Document

1. **Architecture Choice**: Why Option 1 vs Option 2 (or both)?
2. **Feature Extraction**: Why layer 3 for ResNet? Why forward hooks vs separate ResNet?
3. **Fine-tuning Strategy**: Which strategy works best and why?
4. **Loss Function**: How to handle predicting both old and new bbox positions?
5. **Data Augmentation**: What augmentations (if any) are used?
6. **Image Preprocessing**: How to handle different image sizes?
7. **Evaluation Metrics**: Why these specific metrics?

---

## Potential Challenges & Solutions

### Challenge 1: DETR expects single image, we have pairs
- **Solution**: Adapt input to DETR (feature diff or pixel diff)

### Challenge 2: Predicting new bbox positions
- **Solution**: Model outputs bboxes for moved objects in frame 2

### Challenge 3: Handling different image sizes
- **Solution**: Resize to fixed size or use padding

### Challenge 4: Limited data
- **Solution**: Data augmentation, careful train/test split

### Challenge 5: DETR's bipartite matching loss
- **Solution**: Use HuggingFace's implementation, adapt for our format

---

## Next Steps (Immediate Actions)

1. **Fix data_ground_truth_labeller.py paths** - Update to use current directory structure
2. **Run labeller** - Generate matched annotations
3. **Create dataloader.py** - Start with basic dataset class
4. **Research DETR** - Understand HuggingFace Transformers DETR API
5. **Create model skeleton** - Set up basic model files

---

## Notes

- Start simple, iterate: Begin with Option 2 (pixel diff) as it's simpler
- Test frequently: Validate each component before moving to next
- Keep experiments organized: Use clear naming for configs and outputs
- Document as you go: Write docstrings and comments
- Version control: Commit frequently with clear messages

