# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:00:57 2024

@author: hussain
"""

import os
import pandas as pd
import random

import collections.abc
# from pptx import Presentation

path = '../../../../../../../data/AML_classes/'
data_path = 'datasets/raabindata/Original_imgs/'
label_map = {'Basophil': 0, 'Eosinophil': 1, 'Erythroblast': 2, 'Lymphocyte (atypical)': 3, 'Lymphocyte (typical)': 4, 
             'Metamyelocyte': 5, 'Monoblast': 6, 'Monocyte': 7, 'Myeloblast': 8, 'Myelocyte': 9, 'Neutrophil (band)': 10, 
             'Neutrophil (segmented)': 11, 'Promyelocyte': 12, 'Promyelocyte (bilobled)': 13, 'Smudge cell': 14}
img_path, label = [], []
for classes in os.listdir(path):
    for img in os.listdir(os.path.join(path, classes)):
        #img_full_path = os.path.join(os.path.join(data_path, classes), img)
        img_path.append(os.path.join(classes, img).replace("\\","/"))
        label.append(label_map[classes])

metadata = pd.DataFrame()
metadata['image'] = img_path
metadata['label'] = label
metadata = metadata.sample(frac = 1)
#print('count: ', len(metadata)) # count:  18365
train_samples = int(len(metadata)*(80/100))
test_samples = int(len(metadata)*(20/100))
# random.shuffle(metadata)
train_data = metadata[:train_samples]
test_data = metadata[train_samples:]
train_data.to_csv('train/'+"metadata_train.txt", index=False)
test_data.to_csv('test/'+"metadata_test.txt", index=False)
        
        
        
# metadata = pd.DataFrame()
# metadata['image'] = img_path
# metadata['label'] = label

# known_index = metadata.label != "unknown"
# metadata = metadata.loc[known_index,:].reset_index(drop = True)
# metadata.to_csv('train/'+"metadata_train.txt", index=False)

"""
name_to_abbr = {
    "Basophil": "BAS",
    "Erythroblast": "EBO",
    "Eosinophil": "EOS",
    "Smudge cell": "KSC",
    "Lymphocyte (atypical)": "LYA",
    "Lymphocyte (typical)": "LYT",
    "Metamyelocyte": "MMZ",
    "Monoblast": "MOB",
    "Monocyte": "MON",
    "Myelocyte": "MYB",
    "Myeloblast": "MYO",
    "Neutrophil (band)": "NGB",
    "Neutrophil (segmented)": "NGS",
    "Promyelocyte (bilobled)": "PMB",
    "Promyelocyte": "PMO"
}
"""