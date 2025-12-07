# Quick Start Plan - Assignment 3

## Immediate Next Steps (Do First)

### Step 1: Fix & Run Data Labeller
- [ ] Fix paths in `data_ground_truth_labeller.py` (line 182-184)
- [ ] Update to use `data/` directory and `index.txt` in root
- [ ] Run labeller to generate matched annotations
- [ ] Verify output format (2 rows per object: old bbox, new bbox)

### Step 2: Create Basic Dataset Class
- [ ] Create `dataloader.py`
- [ ] Implement `MovedObjectDataset` class
- [ ] Load image pairs and matched annotations
- [ ] Handle image preprocessing (resize, normalize)
- [ ] Test on a few samples

### Step 3: Understand DETR API
- [ ] Research HuggingFace Transformers DETR
- [ ] Understand input/output format
- [ ] Test loading `facebook/detr-resnet-50`
- [ ] Understand model structure (backbone, transformer, heads)

### Step 4: Implement Simple Architecture First (Option 2)
- [ ] Create `models/detr_pixel_diff.py`
- [ ] Implement pixel difference approach
- [ ] Feed to DETR model
- [ ] Test forward pass

### Step 5: Create Training Loop
- [ ] Create `training/trainer.py`
- [ ] Implement basic training loop
- [ ] Handle DETR loss
- [ ] Test on small subset

### Step 6: Add Evaluation
- [ ] Create `evaluation/metrics.py`
- [ ] Implement precision, recall
- [ ] Create `evaluate.py` script

### Step 7: Implement Fine-tuning Strategies
- [ ] Create `training/finetune_strategies.py`
- [ ] Implement all 4 strategies
- [ ] Test each strategy

### Step 8: Implement Option 1 Architecture
- [ ] Create `models/detr_feature_diff.py`
- [ ] Implement feature difference approach
- [ ] Test forward pass

### Step 9: Create Config System
- [ ] Create `configs/` directory
- [ ] Create config files for each experiment
- [ ] Update training script to use configs

### Step 10: Create SLURM Scripts
- [ ] Create `scripts/train.slurm`
- [ ] Create `scripts/eval.slurm`
- [ ] Test job submission

### Step 11: Run Experiments
- [ ] Run all training configurations
- [ ] Collect results
- [ ] Generate visualizations

### Step 12: Documentation
- [ ] Update README
- [ ] Write technical report
- [ ] Organize submission files

---

## Critical Questions to Answer

1. **What does DETR predict?** 
   - We need to predict new bbox positions for moved objects
   - How to format ground truth for DETR?

2. **How to handle old vs new bboxes?**
   - Do we predict both? Or just new positions?
   - Assignment says "detect objects that moved" - likely just new positions

3. **Class labels:**
   - 6 classes: Unknown=0, person=1, car=2, other vehicle=3, other object=4, bike=5
   - DETR needs num_classes parameter

4. **Image preprocessing:**
   - Images may have different sizes
   - Need to resize to fixed size (e.g., 800x800 or 640x640)

5. **Loss function:**
   - DETR uses bipartite matching loss
   - HuggingFace provides this, but need to ensure it works with our format

---

## File Structure to Create

```
assignment_3/
├── models/
│   ├── __init__.py
│   ├── detr_base.py
│   ├── detr_feature_diff.py
│   └── detr_pixel_diff.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── loss.py
│   └── finetune_strategies.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── visualize.py
├── configs/
│   ├── model_config.py
│   ├── train_config.py
│   └── experiments/
│       └── (experiment configs)
├── scripts/
│   ├── train.slurm
│   └── eval.slurm
├── dataloader.py
├── train.py
├── evaluate.py
├── utils.py
└── requirements.txt
```

---

## Key Implementation Details

### Matched Annotation Format
```
match_id_0 x_old y_old w_old h_old type
match_id_0 x_new y_new w_new h_new type
match_id_1 x_old y_old w_old h_old type
match_id_1 x_new y_new w_new h_new type
...
```

### DETR Input Format
- Single image tensor: `[batch, channels, height, width]`
- For Option 1: Feature difference tensor
- For Option 2: Pixel difference image

### DETR Output Format
- `logits`: `[batch, num_queries, num_classes + 1]` (background class)
- `pred_boxes`: `[batch, num_queries, 4]` (normalized cx, cy, w, h)

### Ground Truth Format for DETR
- Need to convert bboxes to normalized format
- Format: `[cx, cy, w, h]` normalized to [0, 1]
- Labels: class indices

---

## Testing Strategy

1. **Unit Tests:**
   - Dataset loading
   - Model forward pass
   - Loss computation
   - Metrics computation

2. **Integration Tests:**
   - Full training loop (1 epoch)
   - Evaluation on small subset

3. **Validation:**
   - Check predictions make sense
   - Visualize sample outputs
   - Compare with ground truth

---

## Time Estimates

- Data preparation: 1-2 days
- Model implementation: 2-3 days
- Training infrastructure: 2-3 days
- Evaluation: 1-2 days
- Experiments: 3-5 days
- Documentation: 2-3 days

**Total: ~2-3 weeks**

