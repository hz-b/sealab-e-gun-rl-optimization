#!/bin/bash
#SBATCH --job-name=optimizers-sweep
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10GB
#SBATCH --partition=main
#SBATCH --array=10-13#,10-13
#SBATCH --spread-job
#SBATCH --exclude=gpu-l40s-1,gpu-a100-1,gpu-a100-5

srun /mnt/work/xfel/bessy/metrix_hackathon/local-venv.sh python3 optimizers_hyperparameters.py $SLURM_ARRAY_TASK_ID

