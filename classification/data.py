import os
from os.path import expanduser
from os.path import join as ospj
import json
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision as tv
from collections import defaultdict
import copy
import h5py
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms as T
import numpy as np
from PIL import Image
import pandas as pd
import random
import torchvision.transforms as tfm
from imageio import imread
from skimage.color import rgb2hsv, hsv2rgb
from data_loader.augmenter import HedLighterColorAugmenter, HedLightColorAugmenter, HedStrongColorAugmenter
import os

from dataset_wbc import DatasetMarr, labels_map, T


from utils import make_dirs
from util_data import (
    SUBSET_NAMES,
    configure_metadata, get_image_ids, get_class_labels,
    GaussianBlur, Solarization,
)

from munch import Munch as mch

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
CLIP_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORM_STD = (0.26862954, 0.26130258, 0.27577711)

def get_transforms(model_type):
    if model_type == "clip":
        norm_mean = CLIP_NORM_MEAN
        norm_std = CLIP_NORM_STD
    elif model_type == "resnet50":
        norm_mean = NORM_MEAN
        norm_std = NORM_STD

    # Train transformations from dataset_wbc.py)
    train_transform = T.Compose([
        T.RandomResizedCrop(size=384, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomApply([T.RandomRotation((0, 180))], p=0.33),
        T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0, saturation=1, hue=0.3)], p=0.33),
        T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1))], p=0.33),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=0.8)], p=0.33),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Test & validation transformations
    test_transform = T.Compose([
        T.Resize(384),  # same as training
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ]) 

    return train_transform, test_transform



class DatasetSynthImage(Dataset):
    def __init__(
        self, 
        synth_train_data_dir, 
        transform, 
        target_label=None, 
        n_img_per_cls=None,
        dataset='matek', 
        n_shot=0,
        real_train_fewshot_data_dir='', 
        is_pooled_fewshot=False, 
        **kwargs
    ):
        self.synth_train_data_dir = synth_train_data_dir
        self.transform = transform
        self.is_pooled_fewshot = is_pooled_fewshot
        
        self.image_paths = []
        self.image_labels = []

        value_counts = defaultdict(int)
        for label, class_name in enumerate(SUBSET_NAMES[dataset]):
            if target_label is not None and label != target_label:
                continue
            for fname in os.listdir(ospj(synth_train_data_dir, class_name)):
                if fname.endswith(".txt"):
                    continue
                if fname.endswith(".json"):
                    continue
                if n_img_per_cls is not None:
                    if value_counts[label] < n_img_per_cls:
                        value_counts[label] += 1
                    else:
                        continue
                self.image_paths.append(
                    ospj(synth_train_data_dir, class_name, fname))
                self.image_labels.append(label)

        if is_pooled_fewshot:
            if n_shot == 0:
                n_shot = 16
            reps = round(n_img_per_cls // n_shot)
            for label, class_name in enumerate(SUBSET_NAMES[dataset]):
                real_img_paths = os.listdir(
                    ospj(real_train_fewshot_data_dir, class_name))
                real_subset = [
                    ospj(
                        real_train_fewshot_data_dir, 
                        class_name, 
                        real_img_paths[i]
                    ) for i in range(n_shot)
                ]
                for i in range(reps):
                    self.image_paths.extend(real_subset)
                    self.image_labels.extend([label] * n_shot)
                
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_label = self.image_labels[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transform(image)
        is_real = "real_train" in image_path

        if self.is_pooled_fewshot:
            return image, image_label, is_real
        else:
            return image, image_label

    def __len__(self):
        return len(self.image_paths)




def get_data_loader(
    dataroot,  # Path principal para DatasetMarr
    dataset_selection="matek",  # Dataset a seleccionar
    bs=32, 
    eval_bs=32,
    is_rand_aug=True,
    model_type=None,
    fold=0,  # Fold para k-fold cross-validation
    is_hsv=True,  # Control de HSV
    is_hed=True,  # Control de HED
):
    # Obtener las transformaciones
    train_transform, test_transform = get_transforms(model_type)

    # Crear el dataset de entrenamiento usando DatasetMarr
    train_dataset = DatasetMarr(
        dataroot=dataroot,
        dataset_selection=dataset_selection,
        labels_map=labels_map,
        fold=fold,
        transform=train_transform if is_rand_aug else test_transform,
        state='train',
        is_hsv=is_hsv,
        is_hed=is_hed,
    )

    # Crear el DataLoader para entrenamiento
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=bs, 
        shuffle=is_rand_aug,
        prefetch_factor=4, 
        pin_memory=True,
        num_workers=8 #16
    )

    # Crear el dataset de validación usando DatasetMarr
    val_dataset = DatasetMarr(
        dataroot=dataroot,
        dataset_selection=dataset_selection,
        labels_map=labels_map,
        fold=fold,
        transform=test_transform,  # same as test
        state='validation',
        is_hsv=is_hsv,
        is_hed=is_hed,
    )

    # Crear el DataLoader para validación
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=eval_bs, 
        shuffle=False, 
        num_workers=8,  # 16
        pin_memory=True
    )

    # Crear el dataset de prueba usando DatasetMarr
    test_dataset = DatasetMarr(
        dataroot=dataroot,
        dataset_selection=dataset_selection,
        labels_map=labels_map,
        fold=fold,
        transform=test_transform,
        state='test',
        is_hsv=is_hsv,
        is_hed=is_hed,
    )

    # Crear el DataLoader para prueba
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=eval_bs, 
        shuffle=False, 
        num_workers=8, #16
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_synth_train_data_loader(
    synth_train_data_dir="data_synth",
    bs=32, 
    is_rand_aug=True,
    target_label=None,
    n_img_per_cls=None,
    dataset='matek',
    n_shot=0,
    real_train_fewshot_data_dir='',
    is_pooled_fewshot=False,
    model_type=None,
):

    train_transform, test_transform = get_transforms(model_type)

    train_dataset = DatasetSynthImage(
        synth_train_data_dir=synth_train_data_dir, 
        transform=train_transform if is_rand_aug else test_transform,
        target_label=target_label,
        n_img_per_cls=n_img_per_cls,
        dataset=dataset,
        n_shot=n_shot,
        real_train_fewshot_data_dir=real_train_fewshot_data_dir,
        is_pooled_fewshot=is_pooled_fewshot,
    ) 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, 
        sampler=None,
        shuffle=is_rand_aug,
        num_workers=8, pin_memory=True, #16
    )
    return train_loader



