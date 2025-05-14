#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:59:33 2025

@author: vilmalehto

A code for actually running the model.
"""
import os

from torch.utils.data import DataLoader

#my classes
from data_preparation import Dataset
from train_model import ModuleTraining

def data_prep(root_folder):
    
    # Create the dataset
    my_data = Dataset(root, image_folder, label_folder, num_classes)
    
    # Wrap it in a DataLoader
    loader = DataLoader(my_data, batch_size=batch_size, shuffle=True)
    
    
    
def module_training(root_path, num_classes):
        #set the file names
        #change these to match your folder structure
        train_img = "train_images"
        train_labels = "train_labels"
        val_img = "val_images"
        val_labels = "val_labels"
    
        #set the model path
        model_path = os.path.join(root_path, "model")
        
        #train and save model to unet.pth file
        ModuleTraining(root_path, train_img, train_labels, val_img, val_labels, num_classes, model_path)
    

if __name__ == "__main__":
    
    #lets just expect that there is a correct folder structure and 
    #ask only the name of the root folder
    print("To run the model give the path to a root folder including the folders for images and model: ")
    root_path = "the user input"
    
    #ask the number of segments
    print("Give the number of segments (including background): ")
    num_of_segments = 5
    
    #does the user want to predict or tain the model
    print(f"Do you want to train({"T"}) or predict({"P"}): ")
    mode = "T"
    
    #set the num of segmentation classes
    num_classes = num_of_segments
    
    if mode ==  "T":
        module_training(root_path, num_classes)

        

