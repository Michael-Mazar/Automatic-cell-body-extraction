# %% Load libraries
import argparse
import random
import shutil
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# %% Setup the function
from skimage.io import imread
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            images.append(imread(img_path))
    return images

imgs = load_images_from_folder(r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_images\images')
# %% TODO: Draw and process images (is this necassery)
for ix, im_name in enumerate(imgs):
        #############Processing###############
        print(im_name)
        img_x = cv2.imread(str(NewTestImages) + im_name)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        img_y = cv2.imread(str(NewTestsMasks) + imgs[ix])
        img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)[:, :, 0:1]

        #############Processing###############
        if len(np.unique(img_y)) > 2:
            print(' restoring {}'.format(im_name))

            ret, img_y = cv2.threshold(img_y, 75, 255, cv2.THRESH_BINARY)

        img_y = img_y.astype(bool)
        img_y = img_y.astype(np.uint8) * 255

        #############Saving in new folder###############
        img_dir = TestImages + '{}'.format(ix + tot_num) + '.tiff'
        mask_dir = TestMasks + '{}'.format(ix + tot_num) + '.tiff'
        plt.imsave(fname=img_dir, arr=np.squeeze(img_x))
        plt.imsave(fname=mask_dir, arr=np.squeeze(img_y), cmap='gray')
