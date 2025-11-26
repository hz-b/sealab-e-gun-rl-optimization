#!/bin/bash
#SBATCH --job-name=sealab-ablation
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10GB
#SBATCH --partition=main
#SBATCH --spread-job
#SBATCH --array=5,6

# Define ablation variants
PARAMS=(
    "layers_3      --layer_size 3"
    "layers_10     --layer_size 10 --learning_rate 1e-5"
    "layers_15     --layer_size 15 --learning_rate 1e-5"
    
    "blow_100      --blow 100"
    "blow_200      --blow 200"
    "blow_300      --blow 300"
    "blow_400      --blow 400"

    "shrink_lin    --shrink_factor lin --learning_rate 1e-5"

    "lr_1e-3       --learning_rate 0.001"
    "lr_1e-4       --learning_rate 1e-4"
    "lr_1e-5       --learning_rate 1e-5"

    "bs_128         --batch_size 128 --learning_rate 1e-5"
    "bs_512        --batch_size 512"

    "patience_250  --patience 250"
    "patience_1000 --patience 1000"

    "decay_0.99    --lr_decay_gamma 0.99"
    "decay_0.95    --lr_decay_gamma 0.95"
)

# Get the current param set
PARAM_LINE="${PARAMS[$SLURM_ARRAY_TASK_ID]}"
NAME=$(echo $PARAM_LINE | cut -d' ' -f1)
ARGS=$(echo $PARAM_LINE | cut -d' ' -f2-)

# Run the training script via srun
srun /mnt/work/xfel/bessy/metrix_hackathon/local-venv.sh python3 surrogate.py $NAME $ARGS

