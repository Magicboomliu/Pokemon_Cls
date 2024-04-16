import os
import sys
sys.path.append("..")
from dataloader.pokemon_loader import PokemonDataSet



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

if __name__=="__main__":
    
    train_dataset = PokemonDataSet(datapath="/data1/su/pokemon/pokemon_dataset/",
                                   trainlist="../filenames/pokemon_train.txt",
                                   vallist="../filenames/pokemon_val.txt",
                                   transforms=None,
                                   mode='train',
                                   scale_size=[224,224])
    
    
    
    for idx, sample in enumerate(train_dataset):

        print(sample['label'])
    
    pass