#loads image-density map pairs and prepare for training/evaluation

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F

class Img_Density_Dataset(Dataset): # a custom PyTorch dataset
    
    def __init__(self, img_list, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        
        if train:
            img_list = img_list*2
            #duplicates dataset (artificially per epoch) to improve stochasticity
            
        if shuffle:
            random.shuffle(img_list)
        #shuffles images list 
        
        self.nSamples =len(img_list)
        self.lines =img_list
        self.transform =transform
        self.train =train
        self.shape = shape
        self.seen=seen
        self.batch_size = batch_size
        self.num_workers=num_workers
        #save everything to class instance
        
        
    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        #load one image+target(densitymap)
        
        img_path = self.lines[index] #select image path by idx
        img,target=load_data(img_path,self.train)
   
        if self.transform is not None:
            img = self.transform(img)
        return img,target