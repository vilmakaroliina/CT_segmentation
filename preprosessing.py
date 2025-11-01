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

if __name__ == "__main__":
    
    image_path = "/Users/vilmalehto/Documents/Koulu/Dippa/To_be_segmented/Images"
    ct_img = nib.load(os.path.join(image_path, "EIT-013_baseline_all.nii.gz"))
    
    #Extract numpy array from nibabel image (clahe works only with 2D formats)
    ct_data = ct_img.get_fdata()
    
    output_path = "/Users/vilmalehto/Documents/Koulu/Dippa/To_be_segmented"
    
    contrast_mod = contrastAdjustment(ct_data, output_path)