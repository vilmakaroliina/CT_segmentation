#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 09:29:51 2025

@author: vilmalehto
"""
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset

class Dataset(Dataset):
    def __init__(self, root_path, images, labels, num_classes, mode="train"):
        
        #could check that image and label path have corresponding files,
        #but let's skip that for now
        
        self.num_classes = num_classes
        self.mode = mode
                
        #get the full data_paths
        self.image_path = os.path.join(root_path, images)
        self.label_path = os.path.join(root_path, labels)
        
        #create sorted list of files in the end of path 
        self.image_files = sorted(os.listdir(self.img_path))
        self.label_files = sorted(os.listdir(self.label_path))
        
        #createas tuple of the files
        pairs = zip(self.image_files, self.label_files)
                
        #create a list of the slices and their corresponding masks
        #(the name of the image file, the name of the label file, and the index for the specific slice)
        self.slices = []
        
        for img_file, label_file in pairs:
            img_vol = nib.load(os.path.join(self.image_path, img_file)).get_fdata()
            #label_vol = nib.load(os.path.join(self.label_path, label_file)).get_fdata()
            
            #could check that the shapes are same, but I am not adding it yet
            
            #loop through the slices
            for i in range(img_vol.shape[-1]):
                self.slices.append((img_file, label_file, i))
                
            
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, index):
        
        #get the infromation
        img_file, label_file, slice_idx = self.slices[index]
        
        #load the 3D volumes
        img_vol = nib.load(os.path.join(self.image_path, img_file)).get_fdata()
        label_vol = nib.load(os.path.join(self.label_path, label_file)).get_fdata()
        
        #get the slices (all pixels/voxels from the specific slice)
        #use np.float32 type to describe the intensity of the voxel/pixel 
        image = img_vol[:, :, slice_idx].astype(np.float32)
        label = label_vol[:, :, slice_idx].astype(np.float32)
        
        #normalize the grayscale values
        image =(image-np.min(image)) / (np.ptp(image) + 1e-8)
        
        # Add channel dimension for image: (1, H, W)
        image = torch.tensor(image).unsqueeze(0)

        # One-hot encode label: (C, H, W) (channels, height, width)
        label = torch.tensor(label, dtype=torch.long)
        label = F.one_hot(label, num_classes=self.num_classes)  # shape: (H, W, C)
        label = label.permute(2, 0, 1).float()  # shape: (C, H, W)

        return image, label
        
            
        