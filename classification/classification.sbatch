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

# Variables de configuración
GPU=0
FOLD=4  # 0, 1, 2, 3, 4, 
DATAROOT=/home/aih/jan.boada/project/codes/classification/csv_files/mix/100
DATASET_SELECTION=matek
OUTPUT_DIR=/home/aih/jan.boada/project/codes/results/classification/clip/mix/prova100/fold$FOLD
CLIP_DIR=/home/aih/jan.boada/project/codes/classification/models/clip_dir
IS_SYNTH_TRAIN=False #True
IS_POOLED_FEWSHOT=False #True
N_CLASS=15

IS_HSV=True
IS_HED=True
IS_RAND_AUG=True
IS_MIX_AUG=True
MODEL_TYPE=clip # resnet50
CLIP_VERSION=ViT-B/32
EPOCHS=100 #10
WARMUP_EPOCHS=30 #3
LR=1e-4
WD=1e-8 #1e-4
MIN_LR=1e-8
LAMBDA_1=0.5 #0.5
BATCH_SIZE=64
BATCH_SIZE_EVAL=1  #16 #8  #1
LOG=tensorboard #wandb

# Mensajes de depuración
echo "=== INICIO DEL JOB ==="
echo "DATAROOT: $DATAROOT"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "BATCH_SIZE_EVAL: $BATCH_SIZE_EVAL"
echo "DATASET_SELECTION: $DATASET_SELECTION"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "FOLD: $FOLD"
echo "IS_SYNTH_TRAIN: $IS_SYNTH_TRAIN"
echo "IS_POOLED_FEWSHOT: $IS_POOLED_FEWSHOT"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "EPOCHS: $EPOCHS"
echo "LR: $LR"
echo "=== EJECUTANDO MAIN.PY ==="

# Ejecutar el script principal
CUDA_VISIBLE_DEVICES=$GPU python main.py \
--model_type=$MODEL_TYPE \
--clip_download_dir=$CLIP_DIR \
--clip_version=$CLIP_VERSION \
--is_lora_image=True \
--is_lora_text=True \
--output_dir=$OUTPUT_DIR \
--dataroot=$DATAROOT \
--dataset_selection=$DATASET_SELECTION \
--is_synth_train=$IS_SYNTH_TRAIN \
--is_pooled_fewshot=$IS_POOLED_FEWSHOT \
--fold=$FOLD \
--n_classes=$N_CLASS \
--lambda_1=$LAMBDA_1 \
--is_hsv=$IS_HSV \
--is_hed=$IS_HED \
--is_rand_aug=$IS_RAND_AUG \
--is_mix_aug=$IS_MIX_AUG \
--epochs=$EPOCHS \
--warmup_epochs=$WARMUP_EPOCHS \
--lr=$LR \
--wd=$WD \
--min_lr=$MIN_LR \
--batch_size=$BATCH_SIZE \
--batch_size_eval=$BATCH_SIZE_EVAL \
--log=$LOG

# Mensaje al finalizar
echo "=== MAIN.PY FINALIZADO ==="
echo "=== FIN DEL JOB ==="