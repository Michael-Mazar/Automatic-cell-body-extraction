import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from cellpose import utils, io, models, dynamics
from cellpose import plot
from glob import glob
from natsort import natsorted
from pathlib import Path
import random
import logging
import napari
import torch
from skimage.measure import regionprops
#%% 
# =============================================================================
# Preprocess fluroscent data file names
# =============================================================================
# %%
# import os
# import shutil

# # Set the path to the directory containing the original files
# source_directory = r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_masks\masks'

# # Set the path to the directory where the renamed files will be saved
# target_directory = r"C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_training_images"

# # Create the target directory if it doesn't exist
# if not os.path.exists(target_directory):
#     os.makedirs(target_directory)

# # Loop through each file in the source directory
# for filename in os.listdir(source_directory):
#     # Check if the current item is a file
#     source_file_path = os.path.join(source_directory, filename)
#     if os.path.isfile(source_file_path):
#         # Split the filename into name and extension
#         name, extension = os.path.splitext(filename)
#         # Create the new filename by adding "mask" before the extension
#         new_filename = f"{name}_mask{extension}"
#         # Construct the full path for the new file in the target directory
#         target_file_path = os.path.join(target_directory, new_filename)
#         # Copy the file to the target directory with the new name
#         shutil.copy2(source_file_path, target_file_path)
#         print(f"Copied '{filename}' and saved as '{new_filename}' in the target directory")


#%% 
# =============================================================================
# Load data for cellpose
# =============================================================================
# %% Load all iamges
from skimage.io import imread
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            images.append(imread(img_path))
    return images

train_dir = r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_training_images'


# imgs = load_images_from_folder(r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_images\images')
# labels = load_images_from_folder(r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_masks\masks')

# get files
output = io.load_train_test_data(train_dir, mask_filter='_mask')
train_data, train_labels, _, test_data, test_labels, _ = output

# Generate random number
random.seed(42)
random_number = random.randint(0, len(train_data))
img = train_data[random_number]
#%% 
# =============================================================================
# Preprocess the image
# =============================================================================
# from PIL import Image

# # Replace 'your_image.tiff' with the path to your TIFF image file
# image_path = r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\drg_cells\2022_10_20_CSU-CY5-50-Pinhole.tif'

# # Load the image
# img = Image.open(image_path)
# img = np.array(img)

# %%
# =============================================================================
# Run Cellpose
# =============================================================================
channels = [0,0]
use_GPU = True
model = models.CellposeModel(gpu=use_GPU, model_type='cyto2')
masks, flows, styles = model.eval(img, 
                                         diameter=30, 
                                         channels=channels)
# masks, flows, styles = model.eval(image,
#                                       diameter=146, #determined during training
#                                       resample = True, # Results in more accurate boundaries
#                                       channels=channels,
#                                       cellprob_threshold = 0,
#                                       flow_threshold = 0.4,
#                                       stitch_threshold = 0,
#                                       min_size = 50)

# =============================================================================
# Examine the results
# =============================================================================
# %%
viewer = napari.Viewer()
image_layer = viewer.add_image(img)
# TODO: Load the label
label_layer = viewer.add_labels(masks)
viewer.add_image(flows[0], name = 'Flows', visible=False)

# Show results in matplotlib as well
fig = plt.figure(figsize=(12,5))
plot.show_segmentation(fig, img, masks, flows[0], channels=[0,0])
plt.tight_layout()
plt.show() 
plt.savefig('cellpose_segmentation_W1P4.png')

fig = plt.figure(figsize=(12,5))
plt.imshow(masks, cmap='gray')


# %% Get properties from masks
def regions_measure(labeled,intensity_image):
    
    """Creates a DataFrame of the positions and measurements of the regions based on a labeled image"""
    
    positions =[]
    for region in regionprops(labeled,intensity_image):
        position = []
        y0, x0, y1, x1 = region.bbox #(min_row, min_col, max_row, max_col)
        r = (x1-x0)/2.0
        y_row = region.centroid[0]
        x_col = region.centroid[1]
        position.append(x_col)
        position.append(y_row)
        position.append(r)
        
        track_window = (x0, y0, x1-x0, y1-y0)
    
        position.append(track_window)
        position.append(region.mean_intensity)

        positions.append(position)
    
    positions  = DataFrame(positions, columns = ['x_col','y_row','r','track_window','mean_intensity'])
    
    return positions



# %%
# =============================================================================
# Retrain the model
# =============================================================================
# TODO: Fix according to https://github.com/jluethi/cellpose-hackaton-liberalilab/blob/main/Cellpose_3D_training_workflow.ipynb
use_GPU = True
channels = [0, 0] # Works when a single input channel is used
n_epochs = 100 # Number of epochs for training
weight_decay = 0.0001
learning_rate = 0.1
min_train_masks=1
test_dir=None
model_name = "retrained_cyto2_flurocells"
with torch.backends.mkldnn.flags(enabled=False):
    # TODO: Switch the full path in save_path to relative path
    new_model_path = model.train(train_data, train_labels, 
                                test_data=test_data,
                                test_labels=test_labels,
                                channels=channels, 
                                save_path=r"C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\models", 
                                n_epochs=n_epochs,
                                learning_rate=learning_rate, 
                                weight_decay=weight_decay,
                                nimg_per_epoch=8,
                                model_name=model_name,
                                SGD=True,
                                min_train_masks=min_train_masks)



# python -m cellpose --train --dir /cellpose_train/ --pretrained_model cyto2 --chan 1 --chan2 2 --img_filter img --mask_filter masks
# %%
