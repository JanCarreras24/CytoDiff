import os
import pandas as pd

# Image directory
root_directory = "/home/aih/jan.boada/project/codes/results/synthetic2/matek/sd2.1/gs2.0_nis50/shot16_seed6_template1_lr0.0001_ep300/train"

data_list = []

# Iterate through each label directory
for label in sorted(os.listdir(root_directory)):
    label_path = os.path.join(root_directory, label)
    if os.path.isdir(label_path):
        for file in sorted(os.listdir(label_path)):
            if file.endswith(('.tiff', '.jpg', '.png', '.jpeg')):
                image_path = os.path.join(label_path, file)
                data_list.append({
                    'image': image_path,
                    'label': label,
                    'dataset': 'matek'
                })

# Convert to DataFrame
df = pd.DataFrame(data_list)

# Save csv file (witout folds, just the base)
csv_path = '/home/aih/jan.boada/project/codes/classification/matek_metadata_base.csv'
df.to_csv(csv_path, index=False)
print(f"CSV base guardado en: {csv_path}")
