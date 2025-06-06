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
conda activate my_env

# Argumentos
NUM_IMAGES=$1
FOLD=$2
GPU=0
DATASET_SELECTION=matek
DATAROOT=/home/aih/jan.boada/project/codes/classification/csv_files/mix/$NUM_IMAGES
OUTPUT_DIR=/home/aih/jan.boada/project/codes/results/classification/clip/mix/$NUM_IMAGES/fold$FOLD
CLIP_DIR=/home/aih/jan.boada/project/codes/classification/models/clip_dir
IS_SYNTH_TRAIN=False # True
IS_POOLED_FEWSHOT=False # True
N_CLASS=15

IS_HSV=True
IS_HED=True
IS_RAND_AUG=True
IS_MIX_AUG=True
MODEL_TYPE=clip # resnet50
CLIP_VERSION=ViT-B/32
EPOCHS=100
WARMUP_EPOCHS=30
LR=1e-4
WD=1e-8
MIN_LR=1e-8
LAMBDA_1=0.5
BATCH_SIZE=64
BATCH_SIZE_EVAL=1
LOG=tensorboard

echo "=== INICIO DEL JOB ==="
echo "DATAROOT: $DATAROOT"
echo "FOLD: $FOLD"
echo "LAMBDA_1: $LAMBDA_1"
echo "NUM_IMAGES: $NUM_IMAGES"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "BATCH_SIZE_EVAL: $BATCH_SIZE_EVAL"
echo "DATASET_SELECTION: $DATASET_SELECTION"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "IS_SYNTH_TRAIN: $IS_SYNTH_TRAIN"
echo "IS_POOLED_FEWSHOT: $IS_POOLED_FEWSHOT"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "EPOCHS: $EPOCHS"
echo "LR: $LR"

mkdir -p $OUTPUT_DIR


echo "=== EJECUTANDO MAIN.PY ==="

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

echo "=== FIN DEL JOB ==="
