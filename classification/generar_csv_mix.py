import os
import pandas as pd
import random

def generate_combined_csv(real_dir, synthetic_dir, max_synthetic_per_class, output_csv_path):
    data_list = []

    # Real images
    for label in sorted(os.listdir(real_dir)):
        label_path = os.path.join(real_dir, label)
        if os.path.isdir(label_path):
            for file in sorted(os.listdir(label_path)):
                if file.endswith(('.tiff', '.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(label_path, file)
                    data_list.append({
                        'image': image_path,
                        'label': label,
                        'dataset': 'matek',
                        'is_real': 1
                    })

    # Synthetic images
    for label in sorted(os.listdir(synthetic_dir)):
        label_path = os.path.join(synthetic_dir, label)
        if os.path.isdir(label_path):
            image_files = [
                f for f in sorted(os.listdir(label_path))
                if f.endswith(('.tiff', '.jpg', '.png', '.jpeg'))
            ]
            # Limit of max synthetic images per class
            selected_files = image_files[:max_synthetic_per_class]

            for file in selected_files:
                image_path = os.path.join(label_path, file)
                data_list.append({
                    'image': image_path,
                    'label': label,
                    'dataset': 'matek',
                    'is_real': 0
                })

    # Convert to dataframe
    df = pd.DataFrame(data_list)

    # Save CSV
    df.to_csv(output_csv_path, index=False)
    print(f"CSV combinado guardado en: {output_csv_path}")

real_images_dir = "/ictstr01/home/aih/jan.boada/project/codes/datasets/data/matek/real_train_fewshot/seed0"
synthetic_images_dir = "/home/aih/jan.boada/project/codes/results/synthetic/matek/sd2.1/gs2.0_nis50/shot16_seed6_template1_lr0.0001_ep300/train"
max_synthetic_per_class = 5000 
output_csv = f'/home/aih/jan.boada/project/codes/classification/csv_files/mix/{max_synthetic_per_class}/matek_metadata_base.csv'

generate_combined_csv(real_images_dir, synthetic_images_dir, max_synthetic_per_class, output_csv)
