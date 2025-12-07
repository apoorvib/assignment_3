# Running on Singularity/HPC Cluster

## Step 1: Build the Singularity Container

On your HPC cluster (you need root/sudo or fakeroot):

```bash
# If you have root access:
sudo singularity build detr.sif detr.def

# If you only have fakeroot (common on HPC):
singularity build --fakeroot detr.sif detr.def

# If neither works, build on your local machine and transfer:
# On your local machine (with Docker):
docker pull pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
singularity build detr.sif docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# Then install packages manually or use overlay
```

**Alternative**: If building containers is not allowed, you can use a pre-built container:

```bash
# Use a pre-built PyTorch container
singularity pull docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
mv pytorch_2.1.0-cuda11.8-cudnn8-runtime.sif detr.sif
```

## Step 2: Install Python Dependencies (if using pre-built container)

If you couldn't build a custom container, install packages in the container:

```bash
# Create an overlay for writable space
singularity exec --nv detr.sif pip install --user transformers torchvision pycocotools
```

Or create a virtual environment:

```bash
# On the HPC login node
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

Then modify `run_singularity.sh` to activate the venv.

## Step 3: Prepare Your Data

Make sure your data is accessible:

```bash
# Check that these exist:
ls data/matched_annotations/
ls index.txt
```

## Step 4: Submit the Job

```bash
# Make the script executable
chmod +x run_singularity.sh

# Submit with default settings (Option 2, 50 epochs)
sbatch run_singularity.sh

# Or customize:
sbatch --export=ARCHITECTURE=1,NUM_EPOCHS=100,BATCH_SIZE=8 run_singularity.sh

# Or edit the script and submit:
nano run_singularity.sh  # Modify parameters
sbatch run_singularity.sh
```

## Step 5: Monitor the Job

```bash
# Check job status
squeue -u $USER

# Monitor output (replace JOBID with your job ID)
tail -f logs/train_JOBID.out

# Check errors
tail -f logs/train_JOBID.err
```

## Alternative: Interactive Testing

Before submitting a full job, test interactively:

```bash
# Request an interactive GPU node
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=32GB --gres=gpu:1 --time=2:00:00 --pty /bin/bash

# Once on the node, run training
singularity exec --nv detr.sif python train.py --architecture 2 --num_epochs 5 --batch_size 2
```

## Singularity Command Reference

### Basic Training Command

```bash
singularity exec --nv detr.sif python train.py \
    --architecture 2 \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --output_dir outputs/exp1
```

### Run with Specific GPU

```bash
CUDA_VISIBLE_DEVICES=0 singularity exec --nv detr.sif python train.py [args]
```

### Run Evaluation

```bash
singularity exec --nv detr.sif python evaluate.py \
    --checkpoint outputs/exp1/best_model.pth \
    --architecture 2
```

## Troubleshooting

### Issue: "CUDA not available"

Make sure you're using `--nv` flag and on a GPU node:

```bash
singularity exec --nv detr.sif python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "Permission denied" or "Cannot write"

Use `--bind` to mount your directories:

```bash
singularity exec --nv --bind $(pwd):/workspace --pwd /workspace detr.sif python train.py [args]
```

### Issue: Container not found

Specify full path:

```bash
singularity exec --nv /path/to/detr.sif python train.py [args]
```

### Issue: Out of memory

Reduce batch size:

```bash
sbatch --export=BATCH_SIZE=2 run_singularity.sh
```

## NYU HPC Specific (Greene)

If you're on NYU Greene cluster:

```bash
# Load Singularity module
module load singularity/3.8.3

# Request GPU resources
sbatch --partition=gpu --gres=gpu:v100:1 run_singularity.sh

# Or for A100:
sbatch --partition=gpu --gres=gpu:a100:1 run_singularity.sh
```

## Configuration Files

You can also create a config file for easier experimentation:

```bash
# Create config.json
cat > config.json << EOF
{
    "architecture": 2,
    "num_epochs": 50,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "finetune_strategy": "full",
    "image_size": 800
}
EOF

# Run with config
singularity exec --nv detr.sif python train.py --config config.json
```
