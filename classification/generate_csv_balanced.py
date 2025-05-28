import os
import pandas as pd
import random

def generate_balanced_csv(real_dir, synthetic_dir, target_per_class, output_csv_path):
    data_list = []

    # Recorremos las clases (se asume que existen en ambas carpetas)
    all_labels = sorted(os.listdir(real_dir))

    for label in all_labels:
        class_real_path = os.path.join(real_dir, label)
        class_synth_path = os.path.join(synthetic_dir, label)

        real_images = []
        synth_images = []

        if os.path.isdir(class_real_path):
            real_images = [
                os.path.join(class_real_path, f)
                for f in sorted(os.listdir(class_real_path))
                if f.lower().endswith(('.tiff', '.jpg', '.png', '.jpeg'))
            ]

        if os.path.isdir(class_synth_path):
            synth_images = [
                os.path.join(class_synth_path, f)
                for f in sorted(os.listdir(class_synth_path))
                if f.lower().endswith(('.tiff', '.jpg', '.png', '.jpeg'))
            ]

        num_real = min(len(real_images), target_per_class)
        num_synth = max(0, target_per_class - num_real)

        selected_real = real_images[:num_real]
        selected_synth = synth_images[:num_synth]

        # Añadir reales
        for img_path in selected_real:
            data_list.append({
                'image': img_path,
                'label': label,
                'dataset': 'matek',
                'is_real': 1
            })

        # Añadir sintéticas
        for img_path in selected_synth:
            data_list.append({
                'image': img_path,
                'label': label,
                'dataset': 'matek',
                'is_real': 0
            })

        print(f"Clase {label}: reales usadas = {len(selected_real)}, sintéticas usadas = {len(selected_synth)}")

    # Guardar CSV
    df = pd.DataFrame(data_list)
    df.to_csv(output_csv_path, index=False)
    print(f"\nCSV balanceado guardado en: {output_csv_path}")

# ==== CONFIGURACIÓN ====
real_images_dir = "/ictstr01/home/aih/jan.boada/project/codes/datasets/data/matek/real_train_fewshot/seed0"
synthetic_images_dir = "/home/aih/jan.boada/project/codes/results/synthetic/matek/sd2.1/gs2.0_nis50/shot16_seed6_template1_lr0.0001_ep300/train"
target_images_per_class = 3000  # puedes cambiar este valor
output_csv = f'/home/aih/jan.boada/project/codes/classification/csv_files/mix/balanced{target_images_per_class}/matek_metadata_balanced.csv'

generate_balanced_csv(real_images_dir, synthetic_images_dir, target_images_per_class, output_csv)
