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

The foulder stucture:
    root-
        |
        |
        - train_img
        |
        |
        - train_labels
        |
        |
        - val_img
        |
        |
        - val_labels
"""

import torch
import os
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet #our unet class
from data_preparation import Dataset #this doesn't exist yet, but the data have to be prepared


class ModuleTraining():
    def __init__(self, root_path, train_img, train_labels, val_img, val_labels, num_classes, model_path):
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
        BATCH_SIZE = 1
        EPOCHS = 2
        #DATA_PATH = "/data" #set the correct path as I get the  data
        # MODEL_SAVE = "/models/unet.pth" #set also this once you have the data 
            #and you can run this, I don't know if it have to be an empty folder
            
        #set the parameters given by user
        self.root_path = root_path
        self.train_images = train_img
        self.train_labels= train_labels
        self.val_images = val_img
        self.val_labels = val_labels
        self.num_classes = num_classes
        #if you  want to save multiple models change the name of the file below
        #inbetween the runs
        self.model_path = os.path.join(model_path, "unet.pth")
        
        
        #use GPU if available
        #device = torch.device("mps" if torch.mps.is_available() else "cpu") #this is mac specific GPU change it to "cuda" for NVIDIA GPU:s
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        #prepare the datasets
        train_set = Dataset(root_path = self.root_path,
                            images = self.train_images,
                            labels = self.train_labels,
                            num_classes = self.num_classes,
                            mode = "train")
      
        val_set = Dataset(root_path = self.root_path,
                          images = self.val_images,
                          labels = self.val_labels,
                          num_classes = self.num_classes,
                          mode = "train")
        
        #create the dataloaders
        train_dataloader = DataLoader(dataset = train_set,
                                      batch_size = BATCH_SIZE,
                                      shuffle = True)
        
        val_dataloader = DataLoader(dataset = val_set,
                                    batch_size  = BATCH_SIZE,
                                    shuffle = True)
        
        #define the model
        #in_channels = 1 for grayscale and 3 for RGB
        #num_classes = 5: background, skull, CSF, brain and haemorrhage
        model = UNet(in_channels = 1, num_classes = self.num_classes).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE) #start with Adam optimizer as we have used that in courses
        
        criterion = nn.BCEWithLogitsLoss() #sets up the loss function, also this 
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
                loss = criterion(y_pred, mask) #comapre the manual segmentation and prediction
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
            print("-"*30) #kuha näitä on enemmmä ku merkkejä tekstissä
            print(f"EPOCH {epoch +1 }")
            print(f"Train loss {train_loss:.4f}")
            print(f"Validation loss {val_loss:.4f}")
        
        #at the end just save the model, no need to return that
        torch.save(model.state_dict(), self.model_path)
    
    
    
    
        
            
    