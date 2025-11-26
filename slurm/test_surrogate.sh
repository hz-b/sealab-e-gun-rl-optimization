#!/bin/bash
#SBATCH --job-name=sealab-nn
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=10GB
#SBATCH --partition=main
#SBATCH --spread-job

        srun /mnt/work/xfel/bessy/metrix_hackathon/local-venv.sh python3 evaluate_surrogate.py
