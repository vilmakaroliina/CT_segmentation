#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 12:59:12 2025

@author: vilmalehto

The code for model training. The code trains the model and saves it at the end 
of the path defined by user. This should be an empty folder.

To train the model you need to know the path for training data and validation 
data. These have to be under the same root folder. 

KEEP THE TESTING DATA IN DIFFERENT FOLDER AND DON'T GIVE THAT PATH TO MODEL HERE!

"""

import torch
import os
import nibabel as nib
import numpy as np
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F


from unet import UNet #our unet class

class HybridLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5, eps=1e-6):
        super(HybridLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.eps = eps
        self.ce_loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, result, target):
        
        # Dice loss
        probs = F.softmax(result, dim=1)
        intersection = (probs * target).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        dice_per_class = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice_per_class.mean()
        
        # Cross-entropy loss
        ce_target = target.argmax(dim=1).long()
        ce_loss = self.ce_loss_fn(result, ce_target)
        
        # Combine
        hybrid_loss = self.weight_dice * dice_loss + self.weight_ce * ce_loss
        return hybrid_loss
    
    
    
class ModuleTraining():
    def __init__(self, train_dataloader, val_dataloader, num_classes, batch_size, model_path, device):
        """
        Functions trains and validates the model. At the end the model is saved
        to a location specified by the user. 

        Parameters
        ----------
        root_path : String
            The path to the root folder.
        train_img : String
            The name of the file containing training images.
        train_labels : String
            The name of the file containing training masks. 
        val_img : String
            The name of the file containing validation images.
        val_labels : String
            The name of the file containing validation masks.
        num_classes : int
            The number of segmented classes, also count background.
        model_path : String
            The path to the file you want to save the model. 

        Returns
        -------
        None.

        """
    
        #set the learning parameters
        LEARNING_RATE = 0.0001
        EPOCHS = 1
        #DATA_PATH = "/data" #set the correct path as I get the  data
        # MODEL_SAVE = "/models/unet.pth" #set also this once you have the data 
            #and you can run this, I don't know if it have to be an empty folder
            
        #set the parameters given by user
        self.train_images = train_dataloader
        self.val_images = val_dataloader
        self.num_classes = num_classes
        #if you  want to save multiple models change the name of the file below
        #inbetween the runs
        self.model_path = os.path.join(model_path, "unet.pth")
        
        
        #define the model
        #in_channels = 1 for grayscale and 3 for RGB
        #num_classes = 5: background, skull, CSF, brain and haemorrhage
        model = UNet(in_channels = 1, num_classes = self.num_classes).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE) #start with Adam optimizer as we have used that in courses
        
        criterion = HybridLoss() #sets up the loss function, also this 
            #is binary loss function -> have to be modified based on the literature
        
        #the training
        for epoch in tqdm(range(EPOCHS)):
            model.train()
            train_running_loss = 0
            
            for idx, img_mask in enumerate(tqdm(train_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                #predict the segments
                y_pred = model(img)
                optimizer.zero_grad() #no idea what this does
                
                #count the loss
                loss = criterion(y_pred, mask) #compare the manual segmentation and prediction
                train_running_loss += loss.item()
                
                #these are also mystery
                loss.backward()
                optimizer.step()
                #but here end the train epoch
                
            #on to validation epoch
            train_loss = train_running_loss / idx+1
            
            model.eval()
            val_runnin_loss = 0
            with torch.no_grad():
                for idx, img_mask in enumerate(tqdm(val_dataloader)):
                    img = img_mask[0].float().to(device)
                    mask = img_mask[1].float().to(device)
                    
                    y_pred = model(img)
                    loss = criterion(y_pred, mask)
                    
                    val_runnin_loss += loss.item()
                    
                val_loss = val_runnin_loss / idx+1
                
            #print for every epoch
            print()
            print("-"*30) #kuha näitä on enemmmä ku merkkejä tekstissä
            print(f"EPOCH {epoch +1 }")
            print(f"Train loss {train_loss:.4f}")
            print(f"Validation loss {val_loss:.4f}")
        
        #at the end just save the model, no need to return that
        torch.save(model.state_dict(), self.model_path)
        
    
    
    
      
class Predicting():
    def __init__(self, images, prediction_path, model, device):
        """
        A class for predicting with the model. 
        
        The images are prepared with custom datset class and wraped into DataLoader.
        The model is set to eval state and the gradient compunation is turned off.
        The program loops through the slices and uses the argmax() to get the class
        with highest likelihood. The prediction and the metadata form the original
        image are saved to corresponding dictionaries. 
        
        After this the program loops through the predictions and saves them in 
        .nii format to file named based on the original image file. 
    
        Parameters
        ----------
        model : PyTorch - CNN network
            The custom UNet model done based on the original UNet archicture 
            (Ronneberger et al., 2015). In this code it is set to handle grayscale 
            images. 
        images:
    
        num_classes : int
            The number of the segmentation classes. 
        prediction_folder : String
            The name of the folder where the predictions will be saved. 
    
        Returns
        -------
        None.
    
        """
        
        #set the model to evaluation state
        model.eval()
        model.to(device)
        
        #createa dictionary for predictions
        volumes = {}
        metadata = {}
        
        #predicting
        with torch.no_grad():
            
            #loop through the slices
            #the image slice, the name of the image file, and the index of the image slice
            for image, img_file, slice_idx in tqdm(images, desc="predict"):
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
                    path = os.path.join(images.dataset.image_path, img_file)
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
            prediction_file = "Prediction_"+img_file
            save_path = os.path.join(prediction_path, prediction_file)
            nib.save(pred_nii, save_path)    
        
            
    