#!/bin/bash

#SBATCH -o /ictstr01/home/aih/jan.boada/project/codes/results/slurm_out_%j.job
#SBATCH -e /ictstr01/home/aih/jan.boada/project/codes/results/slurm_error_%j.job
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH --nice=10000

source $HOME/.bashrc

### Helping Guide : https://bioinformatics_core.ascgitlab.helmholtz-muenchen.de/it_hpc_documentation/index.html 

conda activate my_env

# Variables
GPU=0
SET_SPLIT=1
SPLIT_IDX=15

DATASET="matek"
N_CLS=15
FEWSHOT_SEED="seed6"
N_SHOT=16    # 1, 4, 8, 11 or 16 --> few-shot
NUM_TRAIN_EPOCH=300

START_RANGE=1
END_RANGE=15

# CLASS_IDXS
if [ $SPLIT_IDX -eq $((SET_SPLIT - 1)) ]; then
    FINAL_END_RANGE=$((N_CLS - 1))
else
    FINAL_END_RANGE=$END_RANGE
fi

CLASS_IDXS=($(seq $((START_RANGE - 1)) $((FINAL_END_RANGE - 1))))
echo "CLASS_IDXS: ${CLASS_IDXS[@]}"
echo "START_RANGE: $START_RANGE"
echo "FINAL_END_RANGE: $FINAL_END_RANGE"

# Ejecutar el script para cada clase
for CLASS_IDX in "${CLASS_IDXS[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --dataset=$DATASET \
    --n_template=1 \
    --fewshot_seed=$FEWSHOT_SEED \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=100 \
    --num_train_epochs=$NUM_TRAIN_EPOCH \
    --report_to="tensorboard" \
    --train_text_encoder=True \
    --is_tqdm=True \
    --output_dir=output \
    --n_shot=$N_SHOT \
    --target_class_idx=$CLASS_IDX \
    --resume_from_checkpoint=None
done
