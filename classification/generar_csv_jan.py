# Generate csv for classification, with 5 folds for each image, justusing using train and test

import os
import pandas as pd
import random

# Directorio con las imágenes
data_dir = '/home/aih/jan.boada/project/codes/results/synthetic/matek/sd2.1/gs2.0_nis50/shot16_seed6_template1_lr0.0001_ep300/train'

data = []

for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    
    if os.path.isdir(folder_path):
        label = folder_name
        images = [img for img in os.listdir(folder_path) if img.endswith(('.tiff', '.jpg', '.png', '.jpeg'))]

        random.shuffle(images)

        # Dividir las imágenes en 5 folds de forma equilibrada (puede que algunos tengan una imagen más)
        num_folds = 5
        folds = [[] for _ in range(num_folds)]
        for idx, img_name in enumerate(images):
            fold_idx = idx % num_folds
            folds[fold_idx].append(img_name)

        # Asignar imágenes a train/test para cada fold
        for fold_idx in range(num_folds):
            set_values = ['test' if i == fold_idx else 'train' for i in range(num_folds)]
            for img_name in folds[fold_idx]:
                img_path = os.path.join(folder_path, img_name)
                data.append([img_path, label, 'matek'] + set_values)

# Guardar CSV
df = pd.DataFrame(data, columns=['image', 'label', 'dataset', 'set0', 'set1', 'set2', 'set3', 'set4'])
csv_file = '/home/aih/jan.boada/project/codes/classification/matek_metadata.csv'
df.to_csv(csv_file, index=False)

print(f"CSV creado con éxito: {csv_file}")
