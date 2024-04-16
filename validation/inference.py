import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append("..")

import torchvision.models as models

from utils.file_io import read_img

from dataloader.pokemon_loader import Pokemon_Categories_Dict

import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def Image_DeNormalization(image_tensor):
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    
    image_denorm = image_tensor * image_std + image_mean
    return image_denorm

def Image_Normalization(image_tensor):
    image_mean = torch.from_numpy(np.array(IMAGENET_MEAN)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    image_std = torch.from_numpy(np.array(IMAGENET_STD)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(image_tensor)
    
    image_denorm = (image_tensor - image_mean )/image_std
    return image_denorm




def Inference():
    
    # load the model
    
    ckpt = "../saved_models/model_best.pth"
    
    ckpt_data = torch.load(ckpt)
    
    
    # define the model
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    class_num = 14 # 你的类别数量
    net.fc = nn.Linear(num_ftrs, class_num)

    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    
    net.load_state_dict(ckpt_data['state_dict'])
    
    print("loaded model suceesfully")
    
    net.eval()
    
    
    model_input = "/data1/su/pokemon/pokemon_dataset/Magikarp/00000001.png"
    
    img = read_img(model_input)
    img = img.astype(np.float32)
    
    img = transform.resize(img, [224,224], preserve_range=True)
    
    img_vis = img
    
    img = np.transpose(img, (2, 0, 1))  # [3, H, W]
    img = torch.from_numpy(img) / 255.
    
    img = img.unsqueeze(0)
    
    img_norm = Image_Normalization(img)
    
    
    
    with torch.no_grad():
        
        outputs = net(img_norm)
        
        _, val_predicted = torch.max(outputs.data, 1)
        
        result = Pokemon_Categories_Dict[val_predicted.data.item()]

        
        plt.figure(figsize=(5,5))
        plt.axis('off')
        plt.title("Estimated {}".format(result))
        plt.imshow(img_vis/255)
        plt.savefig("est_result.png")
        
        
        print(result)
        pass
    
    # normaliztion
    
    
    
    
    
    
    
    
    
    
    pass



if __name__=="__main__":
    
    Inference()
    pass