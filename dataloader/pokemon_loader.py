from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset
import os
import sys
sys.path.append("../")


from skimage import io, transform
import numpy as np
from PIL import Image
from utils.file_io import read_img,read_text_lines
from skimage import io, transform


Pokemon_Categories = ['Dewgong', 'Hypno', 'Ninetales', 'Beedrill', 'Magikarp', 'Kadabra', 'Jigglypuff', 
 'Lapras', 'Gastly', 'Ivysaur', 'Farfetchd', 'Articuno', 'Electabuzz', 'Chansey']

Pokemon_Categories_Dict = {0:'Dewgong', 1:'Hypno', 2:'Ninetales', 3:'Beedrill', 4:'Magikarp', 5:'Kadabra', 6:'Jigglypuff', 
 7:'Lapras', 8:'Gastly', 9:'Ivysaur', 10:'Farfetchd', 11:'Articuno', 12:'Electabuzz', 13:'Chansey'}

Pokemon_Categories_Dict_Inverse = {'Dewgong':0, 'Hypno':1, 'Ninetales':2, 'Beedrill':3, 'Magikarp':4, 'Kadabra':5, 'Jigglypuff':6, 
 'Lapras':7, 'Gastly':8, 'Ivysaur':9, 'Farfetchd':10, 'Articuno':11, 'Electabuzz':12, 'Chansey':13}


class PokemonDataSet(Dataset):
    def __init__(self,datapath,
                 trainlist,
                 vallist,
                 transforms,
                 mode='train',
                 scale_size=[224,224]) -> None:
        super().__init__()
        
        self.datapath = datapath
        self.trainlist = trainlist
        self.vallist = vallist
        self.transforms = transforms
        self.mode = mode
        
        self.scale_size = scale_size
        
        
        final_dict = {
            'train': self.trainlist,
            "val": self.vallist}
        
        data_filanmes = final_dict[self.mode]
        
        
        lines = read_text_lines(data_filanmes)
        
        self.samples = []
        
        for line in lines:
            
            sample = dict()
    
            fname = line
            category = fname [:-(len(os.path.basename(fname))+1)]
        
            pokemon_filanems = os.path.join(self.datapath,fname)
            
            
            sample['img'] = pokemon_filanems
            sample['label'] = category
        
            self.samples.append(sample)
            
            
        
        
    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]
        
        
        # read the image.
        sample['img'] = read_img(sample_path['img'])
        # rescale the image with the target size.
        sample['img'] = transform.resize(sample['img'], self.scale_size, preserve_range=True)
        sample['img'] = sample['img'].astype(np.float32)
        
        
        # read the images
        sample['label'] = Pokemon_Categories_Dict_Inverse[sample_path['label']]
        
        

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        
        return sample

    def __len__(self):
        return len(self.samples)