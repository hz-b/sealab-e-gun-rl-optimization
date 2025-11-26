#!/bin/bash
#SBATCH --job-name=sealab-total-test
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 100
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=main
#SBATCH --spread-job
#SBATCH --array=0
#SBATCH --exclude=gpu-l40s-1

        srun /mnt/work/xfel/bessy/metrix_hackathon/local-venv.sh python3 test_model_simulation.py $SLURM_ARRAY_TASK_ID
