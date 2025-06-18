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
N_SET_SPLIT=5
SPLIT_IDX=2

BS=10
NIPC=500
SD="sd2.1"
GS=2.0

N_SHOT=16
N_TEMPLATE=1

MODE="datadream"
DD_LR=1e-4
DD_EP=300

DATASET="matek" # eurosat
IS_DATASETWISE=False
FEWSHOT_SEED="seed6"


CUDA_VISIBLE_DEVICES=$GPU python main.py \
--bs=$BS \
--n_img_per_class=$NIPC \
--sd_version=$SD \
--mode=$MODE \
--guidance_scale=$GS \
--n_shot=$N_SHOT \
--n_template=$N_TEMPLATE \
--dataset=$DATASET \
--n_set_split=$N_SET_SPLIT \
--split_idx=$SPLIT_IDX \
--fewshot_seed=$FEWSHOT_SEED \
--datadream_lr=$DD_LR \
--datadream_epoch=$DD_EP \
--is_dataset_wise_model=$IS_DATASETWISE

# done
