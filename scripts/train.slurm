#!/bin/bash
#SBATCH --job-name=detr_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load modules (adjust based on your cluster)
# module load python/3.8
# module load cuda/11.0

# Activate virtual environment (adjust path)
# source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Create logs directory
mkdir -p logs

# Run training
# Example: python train.py --config configs/experiments/exp1_option2_full_finetune.json
python train.py --config $1

echo "End time: $(date)"

