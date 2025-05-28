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

# from modules.mdlt.utils import misc
# from modules.mdlt.dataset.fast_dataloader import InfiniteDataLoader, FastDataLoader

labels_map = {
        'Basophil': 0,
        'Erythroblast': 1,
        'Eosinophil': 2,
        'Smudge cell': 3,
        'Atypical Lymphocyte': 4,
        'Typical Lymphocyte': 5,
        'Metamyelocyte': 6,
        'Monoblast': 7, 
        'Monocyte': 8,
        'Myelocyte': 9,
        'Myeloblast':10,
        'Band Neutrophil': 11, 
        'Segmented Neutrophil': 12,
        'Promyelocyte Bilobed': 13,
        'Promyelocyte': 14
    }

dataset_image_size = {  
    "Ace_20":250,   #250,
    "matek":345,   #345, 
    "MLL_20":288,   #288,
    "BMC_22":250,   #288,
    }
    
class DatasetMarrMix(Dataset):  # bo
    def __init__(self, 
                 dataroot,
                 dataset_selection,
                 labels_map,
                 fold,
                 transform=None,
                 state='train',
                 is_hsv=False,
                 is_hed=False):
        super(DatasetMarrMix, self).__init__()
        
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
        
        file_path = self.metadata.loc[idx,"image"]
        image = imread(file_path)[:,:,[0,1,2]]
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
        label = self.labels_map[label_name]
        
        if self.is_hsv:
            image = self.colorize(image).clip(0.,1.)
        
        if self.is_hed:
            self.hed_aug.randomize()
            image = self.hed_aug.transform(image)
        
        img = self.to_tensor(copy.deepcopy(image))
        image = self.from_tensor(img)
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label).long()

        # Leer la columna is_real
        is_real = self.metadata.loc[idx, "is_real"]
        is_real = torch.tensor(is_real).long()

        return image, label, is_real

def imshow(img):
    npimg = img.numpy()
    #print('npimg:', npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

if __name__ == '__main__':
    # settings
    dataroot = '../../datasets/'
    batch_size = 4
    fold = 0
    state = 'test'
    is_hsv = False
    is_hed = True
    # resize=224
    # random_crop_scale=(0.8, 1.0)
    # random_crop_ratio=(0.8, 1.2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_transform = T.Compose([
                           T.RandomResizedCrop(size=384, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
                           T.RandomHorizontalFlip(0.5),
                           T.RandomVerticalFlip(0.5),
                           T.RandomApply([T.RandomRotation((0,180))], p=0.33),
                           T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0, saturation=1, hue=0.3)], p=0.33),
                           T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1))], p=0.33), 
                           T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=0.8)], p=0.33),
                           #T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.33),
                           #T.RandomEqualize(p=0.33),
                           #T.RandomApply([T.RandomInvert()], p=0.33),
                           #T.RandomApply([T.RandomGrayscale(p=0.2)], p=0.33),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                           ])
    

    labels_map = {
        'Basophil': 0,
        'Erythroblast': 1,
        'Eosinophil': 2,
        'Smudge cell': 3,
        'Atypical Lymphocyte': 4,
        'Typical Lymphocyte': 5,
        'Metamyelocyte': 6,
        'Monoblast': 7, 
        'Monocyte': 8,
        'Myelocyte': 9,
        'Myeloblast':10,
        'Band Neutrophil': 11, 
        'Segmented Neutrophil': 12,
        'Promyelocyte Bilobed': 13,
        'Promyelocyte': 14
    }
    
    #train_sel = ["BMC_22"]    #["Ace_20", "Mat_19", "MLL_20", "BMC_22"]
    train_sel = ["matek"]    #["Ace_20", "Mat_19", "MLL_20", "BMC_22"]
    
    # trainset
    train_dataset = DatasetMarrMix(dataroot,
                                train_sel,
                                labels_map,
                                fold,
                                train_transform,
                                state,
                                is_hsv,
                                is_hed)
    print('trainset samples:', len(train_dataset))
    trainset_loader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  num_workers=0,
                                  shuffle=True)
    print('#trainset_loader:', len(trainset_loader)) 
    
    # training loop
    for epoch in range(1):
        print('epoch:', epoch)
        for i, (img, label, is_real) in enumerate(trainset_loader):
            print('train::imgs:', img.shape, img.min(), img.max())
            print('train::labels:', label)
            print('train::is_real:', is_real)
            
            # show images
            plt.figure()
            imshow(torchvision.utils.make_grid(img))
            break

    # print('trainset samples:', len(train_dataset))
    # trainset_loader = DataLoader(train_dataset, 
    #                              batch_size=batch_size,
    #                              pin_memory=True,
    #                              num_workers=0,
    #                              shuffle=True)
    # print('#trainset_loader:', len(trainset_loader))
    
    # # testset
    # test_dataset = DatasetMarrMix(dataroot,
    #                       test_sel,
    #                       labels_map,
    #                       test_transform,
    #                       state='test')
    # print('testset samples:', len(test_dataset))
    
    # testset_loader = DataLoader(test_dataset, 
    #                             batch_size=batch_size,
    #                             pin_memory=True,
    #                             num_workers=0,
    #                             shuffle=False)
    
    # print('#testset_loader:', len(testset_loader))
    
    # # dataset details
    # print("###################################################################")
    # print('training labels distribution:')
    # print('total:', len(train_dataset))
    # print(train_dataset.metadata['label'].value_counts())
    # # print("###################################################################")
    # # print('valid labels distribution:')
    # # print('total:', len(val_dataset))
    # # print(val_dataset.dataset.metadata['label'].value_counts())
    # print("###################################################################")
    # print('test labels distribution:')
    # print('total:', len(test_dataset))
    # print(test_dataset.metadata['label'].value_counts())
    # print("###################################################################")
          
    # # training loop
    # for epoch in range(1):
    #     print('epoch:', epoch)
    #     for i, data in enumerate(testset_loader):
    #         img, label, label_name, path  = data['img'], data['label'], data['label_name'], data['path']
            
    #         # img = img.cuda()
    #         # raw_img = raw_img.cuda()
    #         # label = label.cuda()
            
    #         print('train::imgs:', img.shape, img.min(), img.max())
    #         # print('train::raw imgs:', raw_img.shape, raw_img.min(), raw_img.max())
    #         print('train::labels:', label)
    #         print('train::labels_name:', label_name)
    #         print('train::paths:', path)
            
    #         # # onehot encoding
    #         # onehot_encoder = OneHotEncoder(sparse=False)
    #         # integer_encoded = label.numpy()
    #         # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    #         # onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #         # print('train::onehot_labels:', onehot_encoded)
        
    #         # show images
    #         plt.figure()
    #         imshow(torchvision.utils.make_grid(img))
    #         # plt.figure()
    #         # imshow(torchvision.utils.make_grid(raw_img))
    #         break
        
        # # test loop
        # for epoch in range(1):
        #     print('epoch:', epoch)
        #     for i, data in enumerate(testset_loader):
        #         img, label, path  = data['img'], data['label'], data['path']
                
        #         print('test::imgs:', img.shape, img.min(), img.max())
        #         print('test::labels:', label)
        #         print('test::paths:', path)
            
        #         # show images
        #         plt.figure()
        #         imshow(torchvision.utils.make_grid(img/255))
        #         break