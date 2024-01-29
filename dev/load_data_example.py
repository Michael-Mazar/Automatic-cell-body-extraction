
# %% Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.io import imread
from os.path import join
from pathlib import Path
# import stackview
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from glob import glob


# %% Load file
# TODO: Generalize to load from the data folder
# dirpath = 'data/Flurocells/all_images/images'
# filename = 'Mar19bS1C3R1_VLPAGl_200x_y.png'
# filepath = Path(dirpath) / filename # Create the file path using Path
filepath = Path(r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\data\Flurocells\all_images\images\Mar19bS1C1R2_VLPAGr_200x_y.png')
filepath = r"C:\Users\micha\Documents\Projects\Automatic cell body extraction\data\Flurocells\all_images\images\Mar19bS1C1R2_VLPAGr_200x_y.png"
img = imread(filepath)

# %% Load from a particular directory
# get files (during training, test_data is transformed so we will load it again)
train_dir = r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\data\Flurocells\all_images\images'
output = io.load_train_test_data(train_dir, mask_filter='_seg.npy')

# %% Setup napari viewer
import napari
from napari.utils import nbscreenshot
viewer = napari.Viewer()
def screenshot() -> None:
    display(nbscreenshot(viewer))


# %% Preprocessing
# image = human_mitosis()
stackview.insight(img)
# %%
