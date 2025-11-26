#!/bin/bash
#SBATCH --job-name=validity-sweep
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10GB
#SBATCH --partition=main
#SBATCH --array=0-7
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --spread-job
#SBATCH --exclude=gpu-l40s-1

# Default values
DEFAULT_LR=1e-3
DEFAULT_BS=1024
DEFAULT_LAYERS=3
DEFAULT_BLOW=256
DEFAULT_SHRINK="log"

case $SLURM_ARRAY_TASK_ID in
  0)
    NAME="default"
    CMD=""
    ;;

  # Learning rate sweep
  1)
    NAME="lr_1e4"
    CMD="--learning_rate 1e-4"
    ;;
  # Batch size sweep
  2)
    NAME="bs_256"
    CMD="--batch_size 256"
    ;;
  # Layer size sweep
  3)
    NAME="layers_4"
    CMD="--layer_size 4"
    ;;
  4)
    NAME="layers_5"
    CMD="--layer_size 5"
    ;;

  # Blow_to sweep
  5)
    NAME="blow_128"
    CMD="--blow_to 128"
    ;;
  6)
    NAME="blow_512"
    CMD="--blow_to 512"
    ;;

  # Shrink factor sweep
  7)
    NAME="shrink_lin"
    CMD="--shrink_factor lin"
    ;;

esac

echo "Running job $SLURM_ARRAY_TASK_ID with config: $CMD"

srun /mnt/work/xfel/bessy/metrix_hackathon/local-venv.sh python3 validity_classifier.py $NAME $CMD

