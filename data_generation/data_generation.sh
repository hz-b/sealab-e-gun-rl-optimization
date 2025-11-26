#!/bin/bash
#SBATCH --job-name=datagen
#SBATCH --output=slurm.%N.%j.out
#SBATCH --cpus-per-task=10
#SBATCH --array=1-2000%90
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=main

srun /mnt/work/xfel/bessy/metrix_hackathon/local-venv.sh python3 data_generation.py $SLURM_ARRAY_TASK_ID
