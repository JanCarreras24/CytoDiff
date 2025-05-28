import sys
import logging
import random
import atexit
import getpass
import shutil
import time
import os
import yaml
import json
import argparse
from os.path import join as ospj

from util_data import SUBSET_NAMES

_MODEL_TYPE = ("resnet50", "clip")


class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self.log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def str2bool(v):
    if v == "":
        return None
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2none(v):
    if v is None:
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return v


def int2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return int(v)


def float2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return float(v)


def list_int2none(vs):
    return_vs = []
    for v in vs:
        if v is None:
            pass
        elif v.lower() in ('none', 'null'):
            v = None
        else:
            v = int(v)
        return_vs.append(v)
    return return_vs


def set_local(args):
    # Local configuration
    if args.dataset_selection not in SUBSET_NAMES:
        raise ValueError(f"Dataset '{args.dataset_selection}' no es válido. Elige entre {list(SUBSET_NAMES.keys())}.")

    
    args.output_dir = args.output_dir or f"./results/{args.dataset_selection}"
    os.makedirs(args.output_dir, exist_ok=True)



def set_output_dir(args):
    # Configuración del directorio de salida
    model_type = args.model_type
    if model_type == 'clip':
        model_type += args.clip_version

    args.output_dir = ospj(args.output_dir, args.dataset_selection, model_type)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


def set_log(output_dir):
    log_file_name = ospj(output_dir, 'log.log')
    Logger(log_file_name)


def set_follow_up_configs(args):
    set_output_dir(args)
    set_log(args.output_dir)


def get_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model_type', type=str2none, default=None,
                        choices=_MODEL_TYPE)
    parser.add_argument('--clip_version', type=str, default='ViT-B/16')

    # CLIP setting
    parser.add_argument("--is_lora_image", type=str2bool, default=True)
    parser.add_argument("--is_lora_text", type=str2bool, default=True)

    # Añadir el argumento clip_download_dir
    parser.add_argument('--clip_download_dir', type=str, default=None,
                        help='Directorio donde se descargarán los modelos CLIP si no están disponibles localmente')

    # Data (DatasetMarr)
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path principal para DatasetMarr (donde se encuentra matek_metadata.csv)')
    parser.add_argument('--dataset_selection', type=str, default='matek',
                        help='Dataset a seleccionar (por ejemplo, Mat_19)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold para k-fold cross-validation')
    parser.add_argument('--is_hsv', type=str2bool, default=True,
                        help='Activar transformaciones HSV')
    parser.add_argument('--is_hed', type=str2bool, default=True,
                        help='Activar transformaciones HED')
    parser.add_argument('--is_rand_aug', type=str2bool, default=True,
                        help='Activar Random Augmentation durante el entrenamiento')

    # Añadir el argumento is_synth_train
    parser.add_argument('--is_synth_train', type=str2bool, default=False,
                        help='Activar entrenamiento sintético (True o False)')
    
    # Añadir el argumento is_pooled_fewshot
    parser.add_argument('--is_pooled_fewshot', type=str2bool, default=False,
                        help='Acticate different transformation fot synthetic & real (True o False)')    
    
    # Añadir el valor de lambda_1
    parser.add_argument("--lambda_1", type=float2none, default=0.8,
                        help="weight for loss from real/synth data")
    

    # Añadir el argumento n_classes
    parser.add_argument('--n_classes', type=int, default=15,
                        help='Número de clases para el modelo ResNet50')

    # Añadir el argumento is_mix_aug
    parser.add_argument('--is_mix_aug', type=str2bool, default=False,
                        help='Activar mezcla de aumentos durante el entrenamiento (True o False)')

    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=str2bool,
        default=True,
        help="Whether or not to use mixed precision for training.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU. Total batch size is proportional to the number of GPUs.",
    )
    parser.add_argument(
        "--batch_size_eval",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--wd",
        type=float2none,
        default=1e-4,
        help="Weight decay for the SGD optimizer.",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float2none,
        help="Maximum learning rate at the end of linear warmup.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=25,
        type=int,
        help="Number of training epochs for the learning-rate-warm-up phase.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate at the end of training.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output folder to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckpt_freq",
        default=100,
        type=int,
        help="Frequency of intermediate checkpointing.",
    )
    parser.add_argument(
        "--seed",
        default=22,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="Url used to set up distributed training.",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore this argument; No need to set it manually.",
    )

    # wandb args
    parser.add_argument('--log', type=str, default='tensorboard', help='How to log')
    parser.add_argument('--wandb_entity', type=str, default='regaz', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='datadream', help='Wandb project name')
    parser.add_argument('--wandb_group', type=str2none, default=None, help='Name of the group for wandb runs')
    parser.add_argument('--wandb_key', default='<your_api_key_here>', type=str, help='API key for W&B.')

    args = parser.parse_args()

    set_local(args)
    set_follow_up_configs(args)

    return args


