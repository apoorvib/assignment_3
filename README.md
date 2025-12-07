# Assignment 3: Moved Object Detection Using DETR

This repository contains the implementation of a DETR-based pipeline for detecting objects that moved between two frames of surveillance footage.

## Project Structure

```
assignment_3/
├── data/                          # Data directory
│   ├── matched_annotations/      # Generated matched annotations
│   └── visual_matches/           # Visualization outputs
├── models/                        # Model implementations
│   ├── detr_base.py              # Base DETR wrapper
│   ├── detr_pixel_diff.py        # Option 2: Pixel difference architecture
│   └── detr_feature_diff.py      # Option 1: Feature difference architecture
├── training/                      # Training utilities
│   ├── trainer.py                # Training loop
│   └── finetune_strategies.py    # Fine-tuning strategies
├── evaluation/                    # Evaluation utilities
│   ├── metrics.py                # Precision, recall, mAP
│   └── visualize.py             # Visualization tools
├── configs/                      # Configuration files
│   ├── train_config.json         # Default training config
│   └── experiments/             # Experiment-specific configs
├── scripts/                       # SLURM scripts
│   ├── train.slurm               # Training job script
│   └── eval.slurm                # Evaluation job script
├── dataloader.py                  # Dataset class
├── train.py                       # Main training script
├── evaluate.py                    # Main evaluation script
├── utils.py                       # Utility functions
├── data_ground_truth_labeller.py  # Ground truth annotation generator
├── index.txt                      # Data index file
└── requirements.txt               # Python dependencies
```

## Installation

1. Clone the repository and navigate to the project directory.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have PyTorch installed with CUDA support (if using GPU):
```bash
# Visit https://pytorch.org/ for installation instructions
```

## Data Preparation

### Step 1: Generate Matched Annotations

First, run the ground truth labeller to generate matched annotations:

```bash
python data_ground_truth_labeller.py
```

This will:
- Read image pairs from `index.txt`
- Match objects between frames using feature similarity and Hungarian algorithm
- Generate matched annotation files in `data/matched_annotations/`
- Create visualizations in `data/visual_matches/`

**Matched Annotation Format:**
Each matched object has 2 rows:
- Row 1 (old): `match_id x_old y_old w_old h_old type`
- Row 2 (new): `match_id x_new y_new w_new h_new type`

### Step 2: Verify Data

Check that matched annotations were generated:
```bash
ls data/matched_annotations/ | head -5
```

## Training

### Basic Training

Train with default settings:
```bash
python train.py
```

### Training with Configuration File

Train using a specific config:
```bash
python train.py --config configs/experiments/exp1_option2_full_finetune.json
```

### Training with Command Line Arguments

```bash
python train.py \
    --architecture 2 \
    --finetune_strategy full \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --output_dir outputs/my_experiment
```

### Architecture Options

- **Option 1 (Feature Difference)**: `--architecture 1`
  - Extracts features from both images using ResNet
  - Computes feature difference
  - Feeds to transformer block only

- **Option 2 (Pixel Difference)**: `--architecture 2` (default)
  - Computes pixel-wise difference
  - Feeds to entire DETR model

### Fine-tuning Strategies

- `full`: Fine-tune all parameters (except conv for Option 1)
- `conv_only`: Only convolutional block (not for Option 1)
- `classification_head_only`: Only classification head
- `transformer_only`: Only transformer block

### Training on SLURM

Submit a training job:
```bash
sbatch scripts/train.slurm configs/experiments/exp1_option2_full_finetune.json
```

Or modify the SLURM script to pass arguments directly.

## Evaluation

### Basic Evaluation

Evaluate a trained model:
```bash
python evaluate.py \
    --model_path outputs/exp1/best_model.pth \
    --architecture 2 \
    --test_index test_index.txt
```

### Evaluation Options

```bash
python evaluate.py \
    --model_path outputs/exp1/best_model.pth \
    --architecture 2 \
    --test_index test_index.txt \
    --iou_threshold 0.5 \
    --score_threshold 0.5 \
    --num_visualizations 20 \
    --output_dir eval_outputs
```

### Evaluation on SLURM

```bash
sbatch scripts/eval.slurm outputs/exp1/best_model.pth 2
```

## Running Experiments

The assignment requires comparing different fine-tuning strategies. Example experiments:

1. **Option 2 - Full Fine-tuning:**
```bash
python train.py --config configs/experiments/exp1_option2_full_finetune.json
```

2. **Option 2 - Classification Head Only:**
```bash
python train.py --config configs/experiments/exp2_option2_classification_head.json
```

3. **Option 2 - Transformer Only:**
```bash
python train.py --config configs/experiments/exp3_option2_transformer_only.json
```

4. **Option 1 - Full Fine-tuning:**
```bash
python train.py --config configs/experiments/exp4_option1_full_finetune.json
```

## Data Organization

All data are stored in the `data/` folder:

```
data/
├── Pair_S_000001_3503_3563/
│   ├── S_000001_frame_3503.png
│   ├── S_000001_frame_3503.annotation.txt
│   ├── S_000001_frame_3563.png
│   └── S_000001_frame_3563.annotation.txt
└── ...
```

The `index.txt` file contains paths to image pairs:
```
data/Pair_S_000001_3503_3563/S_000001_frame_3503.png,data/Pair_S_000001_3503_3563/S_000001_frame_3503.annotation.txt,data/Pair_S_000001_3503_3563/S_000001_frame_3563.png,data/Pair_S_000001_3503_3563/S_000001_frame_3563.annotation.txt
```

## Object Classes

The dataset has 6 object classes:
- 0: Unknown
- 1: Person
- 2: Car
- 3: Other Vehicle
- 4: Other Object
- 5: Bike

## Output Files

Training outputs:
- `outputs/<experiment>/best_model.pth`: Best model checkpoint
- `outputs/<experiment>/final_model.pth`: Final model checkpoint
- `outputs/<experiment>/history.json`: Training history
- `outputs/<experiment>/config.json`: Training configuration

Evaluation outputs:
- `eval_outputs/results.json`: Evaluation metrics
- `eval_outputs/visualizations/*.png`: Sample visualizations

## Notes

- Images may have different dimensions - the dataloader handles resizing
- The model uses normalized bounding box coordinates [cx, cy, w, h]
- DETR uses bipartite matching for loss computation
- Training history and checkpoints are saved automatically

## Troubleshooting

1. **Out of Memory**: Reduce `batch_size` in config
2. **CUDA errors**: Check GPU availability and PyTorch CUDA installation
3. **Missing matched annotations**: Run `data_ground_truth_labeller.py` first
4. **Import errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)

## Citation

If using DETR, please cite:
```
@article{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  journal={ECCV},
  year={2020}
}
```
