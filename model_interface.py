#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:59:33 2025

@author: vilmalehto

A code for actually running the model.
"""
import os
import torch
import tqdm
import nibabel as nib
import numpy as np

from torch.utils.data import DataLoader

#my classes
from data_preparation import Dataset
from train_model import ModuleTraining
from unet import UNet

    
def predicting(model, device, root_path, image_folder, num_classes, prediction_folder):
    """
    A function for predicting with the model. 

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    root_path : String
        The path to the root folder.
    image_folder : String
        The name of the image folder.
    num_classes : int
        The number of the segmentation classes. 
    prediction_folder : TString
        The name of the folder where the predictions will be saved. 

    Returns
    -------
    None.

    """
    
    #set the model to evaluation state
    model.eval()
    model.to(device)
    
    #Create the dataset of the images
    my_data = Dataset(root_path, image_folder, "None", num_classes, mode = "predict")
    
    #Wrap data in DataLoader
    data_loader = DataLoader(dataset = my_data, 
                             batch_size = 1, 
                             shuffle= True)
    #create folder for results (the segment masks)
    #the folder have to be empty and same name folder cannot exist before hand
    
    #createa dictionary for predictions
    volumes = {}
    metadata = {}
    
    #Predicting
    #turn off gradient computantion, to save memory 
    with torch.no_grad():
        
        #loop through the slices
        #the image slice, the name of the image file, and the index of the image slice
        for image, img_file, slice_idx in tqdm(data_loader, desc="predict"):
            #get the 2D image to gpu if available
            image = image.to(device)
            #pass the image through the model
            output = model(image)
            #get the prediction
            #argmax() gets the most likely class per pixel/voxel
            pred = torch.argmax(output, dim=1).numpy().squeeze(0) #2D numpy array
            
            #extract the filename and slice index
            img_file = img_file[0]
            slice_idx = slice_idx.item()
            
            #check if the file already exist in volumes dict
            if img_file not in volumes:
                #if not create it
                volumes[img_file] = {}
                
                #get the metadata from original file
                path = os.path.join(my_data.image_path, img_file)
                nii = nib.load(path)
                metadata[img_file] = (nii.affine, nii.header)
                
            #save the prediction to correct spot
            volumes[img_file][slice_idx] = pred
                #volumes = { "img_file_1.nii": {0: pred, 1: pred}, 
                #           "img_file_2.nii" = {0: pred, 1: pred}}
            
    #loop through the predictions
    for img_file, slice_preds in volumes.items():
        #Find the highest slice number for the specific image file
        max_idx = max(slice_preds.keys()) + 1
        #initialize the array for 3D label
        volume = np.zeros((list(slice_preds.values())[0].shape[0],
                           list(slice_preds.values())[0].shape[1],
                           max_idx), dtype=np.uint8)
        
        #get each prediction slice
        for idx, prediction in slice_preds.items():
            #save the slice to its correct spot in the 3D array
            volume[:, :, idx] = prediction
        
        #get the corresponding metadata
        affine, header = metadata[img_file]
        #create the .nii format with prediction volume and metadata
        pred_nii = nib.Nifti1Image(volume, affine, header)
        
        #save the prediction to file corresponding to image file
        prediction_file = "Pred"+img_file
        save_path = os.path.join(root_path, prediction_folder, prediction_file)
        nib.save(pred_nii, save_path)      
    
    
    
def module_training(root_path, num_classes, device):
    #set the file names
    #change these to match your folder structure
    train_img = "train_images"
    train_labels = "train_labels"
    val_img = "val_images"
    val_labels = "val_labels"

    #set the model path
    model_path = os.path.join(root_path, "model")
    
    #train and save model to unet.pth file
    ModuleTraining(root_path, train_img, train_labels, val_img, val_labels, num_classes, model_path, device)
    
    

if __name__ == "__main__":
    
    #lets just expect that there is a correct folder structure and 
    #ask only the name of the root folder
    root_path = input("To run the model give the path to a root folder including the folders for images and model: ")
    
    #ask the number of segments
    num_classes = input("Give the number of segments (including background): ")
    #change these commentation if you want to
    #num_classes = 5
    
    #does the user want to predict or tain the model
    mode = input("Do you want to train(T) or predict(P): ")
    
    if mode != "T" or "P":
        mode = input("The mode have to be T for train or P for predcit: ")

    #find the gpu if available
    #define the device, gpu/cpu
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    
    if mode ==  "T":
        module_training(root_path, num_classes, device)
        
    else:
        #if predicting we need information about the model and place for predictions
        
        #I will set these myself, don't want to be constantly giving these as input
        #weights_path = input("Give the name of the folder where model weights are saved: ")
        weights_path = "path"
        
        #prediction_folder = input("Give the name of the folder where predictions will be saved: ")
        prediction_folder = "predictions"
        
        #image_folder = input("Give the name of the folder where the images are: ")
        image_folder = "test_images"
        
        #load the model
        model = UNet()
        model.load_state_dict(torch.load(weights_path, map_location = device))
        
        predicting(model, device, root_path, image_folder, num_classes, prediction_folder)
        

