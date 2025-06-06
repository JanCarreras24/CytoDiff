#!/bin/bash
#SBATCH -o /ictstr01/home/aih/jan.boada/project/codes/results/fid_out_%j.log
#SBATCH -e /ictstr01/home/aih/jan.boada/project/codes/results/fid_err_%j.log
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH --nice=10000

# Activar entorno
source $HOME/.bashrc
conda activate my_env

# Mensajes de control
echo "=== INICIO DE CÁLCULO DE FID ==="
date
nvidia-smi

# Ejecutar script
python fid_dataset.py

echo "=== FIN DE FID ==="
date
