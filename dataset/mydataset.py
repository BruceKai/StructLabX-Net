import glob

import numpy as np
import rasterio as rio
import torch
from rasterio.enums import Resampling
from rasterio import features
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from skimage import  morphology as mp,segmentation as seg
from dataset.data_augmentation import DataAug
from skimage.transform import rescale
from skimage import feature
import cv2 as cv

transform = DataAug()

def detect_edge(image):
    dy = cv.Sobel(image, cv.CV_64F,0,1)
    dx = cv.Sobel(image, cv.CV_64F,1,0)
    edge_intensity = np.sqrt(dy**2+dx**2)
    return edge_intensity   

def image_linear_transform(image, scale=[2,98]):
    image = np.float32(image)
    for i in range(image.shape[0]):
        min_,max_ = np.percentile(image[i],scale)
        image[i] = (image[i]-min_)/(max_-min_+1e-7)
    return np.clip(image,0,1)
       
class MyDataset(Dataset):
    def __init__(self,root_dir,data_aug=False):
        super(MyDataset, self).__init__()
        img_path=glob.glob(root_dir+'/img/*.tif')
        lbl_path=glob.glob(root_dir+'/lbl/*.tif')
        img_path.sort()
        lbl_path.sort()

        self.img_path = img_path
        self.lbl_path = lbl_path
        self.data_aug = data_aug

                
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        
        img = rio.open(self.img_path[index])
        lbl = rio.open(self.lbl_path[index])
        row = np.random.randint(15,50)
        col = np.random.randint(15,50)
        
        # random-crop 
        img = img.read(window=((row,row+64),(col,col+64))) if self.data_aug else img.read(window=((32,96),(32,96)))
        target = lbl.read(window=((row,row+64),(col,col+64))) if self.data_aug else lbl.read(window=((32,96),(32,96)))
        # The target tensor has a shape of (2, h, w). The first channel represents the class attribute of each pixel, 
        # with values ranging from [0, k-1]. The second channel has binary values (0 or 1) indicating which pixels are labeled.
        
        target = target[-2:,:,:]
        
        # img = img.read()
        img = img/10000.
        img[np.isnan(img)] = 0
        img[(img<0)|(img>1)] = 0
        
        
        if self.data_aug:
            # random filp and random rotate
            input_data = dict({'image':img,
                               'label':target})
            input_data = transform.transform(input_data)
            img = input_data['image'][0].copy()
            target = input_data['label'].copy() 


        edge_intensity = np.stack([detect_edge(img[i]) for i in range(img.shape[0])],axis=0)
        edge_intensity = image_linear_transform(edge_intensity,[2,98])
        edge_intensity = np.mean(edge_intensity,0)
        edge_intensity[np.isnan(edge_intensity)] = 0
        edge_intensity[(edge_intensity<0)|(edge_intensity>1)] = 0

        # (t*c,h,w) -> (t,c,h,w)
        img = img.reshape(12,4,img.shape[1],img.shape[2])

        return torch.from_numpy(img),torch.from_numpy(edge_intensity),torch.from_numpy(target)
