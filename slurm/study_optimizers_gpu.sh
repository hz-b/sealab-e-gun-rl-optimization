#!/bin/bash
#SBATCH --job-name=optimizers-sweep
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=300GB
#SBATCH --partition=main
#SBATCH --array=3#6-9#-9#3-9
#SBATCH --spread-job
#SBATCH --exclude=gpu-l40s-1,gpu-a100-1,gpu-a100-5

srun /mnt/work/xfel/bessy/metrix_hackathon/local-venv.sh python3 optimizers_hyperparameters.py $SLURM_ARRAY_TASK_ID

