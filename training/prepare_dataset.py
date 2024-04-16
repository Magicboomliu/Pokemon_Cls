import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from torch.utils.data import DataLoader

from dataloader.pokemon_loader import PokemonDataSet
from dataloader import transforms

import numpy as np
from tqdm import tqdm
from utils.devtools import Convert_IMGTensor_To_Numpy
import matplotlib.pyplot as plt



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def Image_DeNormalization(image_tensor):
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    
    image_denorm = image_tensor * image_std + image_mean
    return image_denorm




# Get Dataset Here
def prepare_dataset(datapath,
                    trainlist,
                    vallist,
                    scale_size,
                    batch_size,
                    test_size,
                    datathread
                    ):
    
    train_transform_list = [transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    train_transform = transforms.Compose(train_transform_list)
    
    
    val_transform_list = [transforms.ToTensor(),
                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    
    val_transform = transforms.Compose(val_transform_list)
    

    train_dataset = PokemonDataSet(datapath=datapath,
                                   trainlist=trainlist,
                                   vallist=vallist,
                                   
                                   mode='train',
                                   scale_size=scale_size,
                                   transforms = train_transform)

    test_dataset = PokemonDataSet(datapath=datapath,
                                   trainlist=trainlist,
                                   vallist=vallist,
 
                                   mode='val',
                                   scale_size=scale_size,
                                   transforms = val_transform)
    
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = test_size, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    num_batches_per_epoch = len(train_loader)
    
    
    return train_loader,test_loader,num_batches_per_epoch



if __name__=="__main__":
    train_loader,test_loader,num_batches_per_epoch = prepare_dataset(datapath="/data1/su/pokemon/pokemon_dataset/",
                                                                     trainlist="../filenames/pokemon_train.txt",
                                                                     vallist="../filenames/pokemon_val.txt",
                                                                     scale_size=[224,224],
                                                                     batch_size=8,
                                                                     test_size=1,
                                                                     datathread=4)

    for idx, sample in enumerate(train_loader):
        print(sample['img'].shape)
        print(sample['label'].shape)
    
    pass

