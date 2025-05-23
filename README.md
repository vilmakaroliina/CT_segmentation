# CT_segmentation
My master thesis code for segmenting head CT images. 

The program is UNet based segmentation program. The UNet model is done based on the original architecture by Ronneberger et al., 2015. 

To run the code you will need libraries:
  os
  torch
    torch.nn
    torch.utils
  tqdm
  nibabel
  numpy

To run the proram you need the grayscale images and corresponding segmentation masks for training. The filestructure must also follow the assumptions made in code. The code assumes that you have one root folder, which includes a folder for training images, training lables, validation images, validation labels and also a folder for the images to be predicted and folder for results. You can save the trained model outside the root folder and also if you use pre-existing weights those can be saved outside the root folder, as full paths can be set for those.

