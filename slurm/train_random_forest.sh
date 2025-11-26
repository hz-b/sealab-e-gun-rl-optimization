#!/bin/bash
#SBATCH --job-name=sealab-forest
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task 40
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=main
#SBATCH --spread-job
#SBATCH --exclude=gpu-l40s-1

        srun /mnt/work/xfel/bessy/metrix_hackathon/local-venv.sh python3 random_forest.py 
