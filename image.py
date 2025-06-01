import random
import os
from PIL import Image,ImageFilter,ImageDraw  # image handling
import numpy as np
import h5py # for loading h5 density maps
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'generated-h5')
    
    img = Image.open(img_path).convert('RGB') # open and convert image to RGB
    gt_file = h5py.File(gt_path) # open corresponding density map file
    target = np.asarray(gt_file['density']) # load density map as numpy array
    
    if False:
        crop_size = (img.size[0]/2,img.size[1]/2)
        if random.randint(0,9)<= -1: 
            
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy)) # crop image
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx] # crop density map
        
        
        if random.random()>0.8:
            target = np.fliplr(target) # horizontal flip of density map
            img = img.transpose(Image.FLIP_LEFT_RIGHT) # horizontal flip of image
    
    # Downsample density map and scale (for smaller input to the model)
    target = cv2.resize(target,(target.shape[1]//8,target.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
    
    return img,target # return processed image and density map
    