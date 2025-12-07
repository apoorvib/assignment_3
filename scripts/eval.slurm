#!/bin/bash
#SBATCH --job-name=detr_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

# Run evaluation
# Example: python evaluate.py --model_path outputs/exp1/best_model.pth --architecture 2
python evaluate.py --model_path $1 --architecture $2 --test_index test_index.txt

echo "End time: $(date)"

