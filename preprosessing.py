#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 09:36:46 2025

@author: vilmalehto
"""
import cv2
import numpy as np
import os
import nibabel as nib

def contrastAdjustment(image, output_path):
    clahe = cv2.createCLAHE(clipLimit = 3.5)
    
    enhanced_volume = np.zeros_like(image)
    #apply clahe slice by slice
    for i in range(image.shape[2]):
        current_slice = image[:, :, i]
        slice_norm = cv2.normalize(current_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        enhanced_volume[:, :, i] = clahe.apply(slice_norm)
    
    output_file = os.path.join(output_path, "Enhanced_EIT-013.V1.1.nii.gz")
    enhanced_nii = nib.Nifti1Image(enhanced_volume, affine=np.eye(4))
    nib.save(enhanced_nii, output_file)
    
    return

def volumeAugmentation(image):
    #set the limits for augmentations
    flip_probability = 0.5
    zoom_range = 0.05
    rotation_range=5
    
    #getthedimensions of image volume
    h, w, d = image.shape
    
    #set the augmentation parameters
    do_flip = np.random.rand() < flip_probability
    zoom_factor = 1 + np.random.uniform(-zoom_range, zoom_range)
    angle = np.random.uniform(-rotation_range, rotation_range)
    
    zoom = cv2.getRotationMatrix2D((w/2, h/2), 0, zoom_factor)
    rotation = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    
    #init the augmented volume
    augmented = np.zeros_like(image)
    
    #apply augmentation for all slices in the image volume
    #all will be zoomed and rotated, but the flipping will happen only in random cases
    for i in range(d):
        current_slice = image[:, :, i]
        
        if do_flip:
            current_slice = cv2.flip(current_slice, 1)
        current_slice = cv2.warpAffine(current_slice, zoom, (w, h), flags=cv2.INTER_LINEAR)
        current_slice = cv2.warpAffine(current_slice, rotation, (w, h), flags = cv2.INTER_NEAREST)
        
        augmented[:, :, i] = current_slice
        
    return augmented
    

if __name__ == "__main__":
    
    image_path = "/Users/vilmalehto/Documents/Koulu/Dippa/To_be_segmented/Images"
    ct_img = nib.load(os.path.join(image_path, "EIT-013_baseline_all.nii.gz"))
    
    #Extract numpy array from nibabel image (clahe works only with 2D formats)
    ct_data = ct_img.get_fdata()
    
    output_path = "/Users/vilmalehto/Documents/Koulu/Dippa/To_be_segmented"
    
    contrast_mod = contrastAdjustment(ct_data, output_path)
    
    
    #call for data augmentation
    for fname in os.listdir(image_path):
        fpath=os.path.join(image_path, fname)
        
        nii =nib.load(fpath)
        volume = nii.get_fdata()
        
        augmented = volumeAugmentation(volume)
        
        output_file = os.path.join(output_path, f"aug_{fname}")
        nib.save(nib.Nifti1Image(augmented, affine=nii.affine, header = nii.header), output_file)
        
        
    
    