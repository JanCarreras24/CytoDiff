import torch
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
from data import get_data_loader

train_loader, val_loader, test_loader = get_data_loader(
    dataroot="/home/aih/jan.boada/project/codes/classification",
    dataset_selection="matek",
    bs=32,
    eval_bs=32,
    is_rand_aug=True,
    model_type="resnet50",
    fold=0,
    is_hsv=True,
    is_hed=True,
)

print(f"Train loader size: {len(train_loader.dataset)}")
print(f"Validation loader size: {len(val_loader.dataset)}")
print(f"Test loader size: {len(test_loader.dataset)}")




'''
# from modules.mdlt.utils import misc
# from modules.mdlt.dataset.fast_dataloader import InfiniteDataLoader, FastDataLoader

labels_map = {
        'Basophil': 0,
        'Eosinophil': 1,
        'Erythroblast': 2,
        'Atypical Lymphocyte': 3,
        'Typical Lymphocyte': 4,
        'Metamyelocyte': 5,
        'Monoblast': 6,
        'Monocyte': 7, 
        'Myeloblast': 8,
        'Myelocyte': 9,
        'Band Neutrophil':10,
        'Segmented Neutrophil': 11, 
        'Promyelocyte': 12,
        'Promyelocyte Bilobed': 13,
        'Smudge cell': 14
    }

dataset_image_size = {  
    "Ace_20":250,   #250,
    "matek":345,   #345, 
    "MLL_20":288,   #288,
    "BMC_22":250,   #288,
    }
    
class DatasetMarr(Dataset):  # bo
    def __init__(self, 
                 dataroot,
                 dataset_selection,
                 labels_map,
                 fold,
                 transform=None,
                 state='train',
                 is_hsv=False,
                 is_hed=False):
        super(DatasetMarr, self).__init__()
        
        self.dataroot = os.path.join(dataroot, '')  

        metadata_path = os.path.join(self.dataroot, 'matek_metadata.csv')
        try:
            metadata = pd.read_csv(metadata_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No hi ha cap csv file a: {metadata_path}")

        set_fold = "kfold" + str(fold)  # Adaptation for the csv file
        if isinstance(dataset_selection, list):
            dataset_index = metadata.dataset.isin(dataset_selection)
        else:
            dataset_index = metadata["dataset"] == dataset_selection
        print(f"Filas que hi ha en total ({dataset_selection}): {dataset_index.sum()}")

        # Filter by fold
        if state == 'train':
            dataset_index = dataset_index & metadata[set_fold].isin(["train"])
        elif state == 'validation':
            dataset_index = dataset_index & metadata[set_fold].isin(["val"])
        elif state == 'test':
            dataset_index = dataset_index & metadata[set_fold].isin(["test"])
        else:
            raise ValueError(f"Estado desconegut: {state}")
        print(f"Filas després de filtrar per fold ({set_fold}, {state}): {dataset_index.sum()}")

        dataset_index = dataset_index[dataset_index].index
        metadata = metadata.loc[dataset_index, :]
        self.metadata = metadata.copy().reset_index(drop=True)
        self.labels_map = labels_map
        self.transform = transform
        self.is_hsv = is_hsv and random.random() < 0.33
        self.is_hed = is_hed and random.random() < 0.33
        
        self.hed_aug = HedLighterColorAugmenter()
        
        # numpy --> tensor
        self.to_tensor = tfm.ToTensor()
        # tensor --> PIL image
        self.from_tensor = tfm.ToPILImage()

    def __len__(self):
        return len(self.metadata)
    
    def read_img(self, path):
        img = Image.open(path)
        if img.mode == 'CMYK':
            img = img.convert('RGB')    
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return img
    
    def colorize(self, image):
        """ Add color of the given hue to an RGB image.
    
        By default, set the saturation to 1 so that the colors pop!
        """
        hue = random.choice(np.linspace(-0.1, 0.1))
        saturation = random.choice(np.linspace(-1, 1))
        
        # hue = random.choice(np.linspace(0, 1))
        # saturation = random.choice(np.linspace(0, 1))
        # print(f"Valor de hue generado en colorize: {hue}")
        # print(f"Valor de saturation generado en colorize: {saturation}")
        hsv = rgb2hsv(image)
        hsv[:, :, 1] = saturation
        hsv[:, :, 0] = hue
        return hsv2rgb(hsv)            

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        dataset =  self.metadata.loc[idx,"dataset"]
        crop_size = dataset_image_size[dataset]
        #print('crop_size:', crop_size)
        
        file_path = self.metadata.loc[idx,"image"]
        #file_path = os.path.join('../../', self.metadata.loc[idx,"image"])
        #image= self.read_img(file_path)
        image= imread(file_path)[:,:,[0,1,2]]
        h1 = (image.shape[0] - crop_size) /2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) /2
        h2 = int(h2)
        
        w1 = (image.shape[1] - crop_size) /2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) /2
        w2 = int(w2)
        image = image[h1:h2,w1:w2, :]
        
        label_name = self.metadata.loc[idx,"label"]
        # print(f"Etiqueta obtenida: {label_name} (tipo: {type(label_name)})")
        label = self.labels_map[label_name]
        
        if self.is_hsv:
            image = self.colorize(image).clip(0.,1.)
            #print('img hsv:', image.shape, image.min(), image.max())
        
        if self.is_hed:
            self.hed_aug.randomize()
            image = self.hed_aug.transform(image)
            #print('img hed:', image.shape, image.min(), image.max())
        
        img = self.to_tensor(copy.deepcopy(image))
        #print('img tensor:', img.shape, img.min(), img.max())
        image = self.from_tensor(img)
        #print('img PIL:', image.size)
        
        if self.transform:
            image = self.transform(image)
            # raw_image = self.transform(raw_image)
        
        label = torch.tensor(label).long()
        
        return image, label
        #return {'img': image, 'label': label, 'label_name': label_name, 'path': file_path}











dataroot = "/home/aih/jan.boada/project/codes/classification"
dataset_selection = "matek"  

train_dataset = DatasetMarr(dataroot, dataset_selection, labels_map, fold=0, state='train')
val_dataset = DatasetMarr(dataroot, dataset_selection, labels_map, fold=0, state='validation')
test_dataset = DatasetMarr(dataroot, dataset_selection, labels_map, fold=0, state='test')

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
'''