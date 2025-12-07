#!/bin/bash
#SBATCH --job-name=detr_training
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Set environment variables
export SINGULARITY_IMAGE="detr.sif"

# Default training arguments (modify as needed)
ARCHITECTURE=${ARCHITECTURE:-2}  # 1=feature diff, 2=pixel diff
NUM_EPOCHS=${NUM_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
IMAGE_SIZE=${IMAGE_SIZE:-800}
FINETUNE_STRATEGY=${FINETUNE_STRATEGY:-full}

# Data paths (modify for your setup)
DATA_DIR="."
MATCHED_ANNOTATIONS_DIR="data/matched_annotations"
INDEX_FILE="index.txt"

# Output directory
OUTPUT_DIR="outputs/arch${ARCHITECTURE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Training configuration:"
echo "  Architecture: $ARCHITECTURE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run training with Singularity
singularity exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    $SINGULARITY_IMAGE \
    python train.py \
    --architecture $ARCHITECTURE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --image_size $IMAGE_SIZE \
    --finetune_strategy $FINETUNE_STRATEGY \
    --data_dir $DATA_DIR \
    --matched_annotations_dir $MATCHED_ANNOTATIONS_DIR \
    --index_file $INDEX_FILE \
    --output_dir $OUTPUT_DIR \
    --save_every 10

echo ""
echo "Training completed at: $(date)"
