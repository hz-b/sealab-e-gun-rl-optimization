#!/bin/bash
#SBATCH --job-name=sealab-dm-study
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10GB
#SBATCH --partition=main
#SBATCH --spread-job
#SBATCH --array=15,16#0-16%6  # 17 experiments

PARAMS=(
    "Reference       "
    "ScaledSigmoid   --model.last_activation sigmoid"
    "Lin_4           --model.layer_size 4 --model.shrink_factor lin"
    "Lin_5           --model.layer_size 5 --model.shrink_factor lin"
    "Log_4           --model.layer_size 4 --model.shrink_factor log"
    "Mish            --model.activation mish"
    "Small           --model.neuron_factor 200"
    "Big             --model.neuron_factor 1000"
    "BatchNorm       --model.batch_norm True"
    "BS_16           --data.batch_size 16"
    "BS_64           --data.batch_size 64"
    "LR_1e-3         --model.learning_rate 0.001"
    "LR_1e-5         --model.learning_rate 1e-5"
    "L2              --model.loss_norm l2"
    "AdamW           --model.optimizer adam_w"
    "Plat_3          --model.lr_scheduler plateau --model.patience 3"
    "Plat_5          --model.lr_scheduler plateau --model.patience 5"
)

# Get the current param line
PARAM_LINE="${PARAMS[$SLURM_ARRAY_TASK_ID]}"
NAME=$(echo "$PARAM_LINE" | cut -d' ' -f1)
ARGS=$(echo "$PARAM_LINE" | cut -d' ' -f2-)

# Run the training with CustomCLI
srun /mnt/work/xfel/bessy/metrix_hackathon/local-venv.sh \
    python3 model.py fit --wandb_name "$NAME" $ARGS
