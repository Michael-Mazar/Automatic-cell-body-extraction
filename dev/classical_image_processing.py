# %% Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.io import imread
from skimage.feature import blob_dog, blob_log
from os.path import join
import os
import cv2
import napari

# =============================================================================
# Load the images and masks into napari
# =============================================================================
# %%
from skimage.io import imread
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            images.append(imread(img_path))
    return images

imgs = load_images_from_folder(r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_images\images')
masks = load_images_from_folder(r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_masks\masks')

# Start with a test image
img = imgs[0]

# Load to napari
viewer = napari.Viewer()
image_layer = viewer.add_image(img)

# =============================================================================
# Enhance
# =============================================================================
# %%
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # tileGridSize=(8,8)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = gray.astype(np.uint16)
cl_img = clahe.apply(gray)
cl_img_color = cv2.cvtColor(cl_img, cv2.COLOR_GRAY2RGB) # Convert back to RGB
cl_img_layer = viewer.add_image(cl_img) # Add to napari

# =============================================================================
# Gaussian blur
# =============================================================================
# %% Try Gaussian blur instead?
sigma = 6
# gaussian_blur_img = ndi.gaussian_filter(cl_img, sigma) # alternative implementation
blurred_img = cv2.GaussianBlur(cl_img,(9,9),-1)
blurred_layer = viewer.add_image(blurred_img) # Add to napari

# =============================================================================
# Adaptive thresholding
# =============================================================================
# %% Try adaptive thresholding afterwards
# FIXME: Threshold is not working well
centers, segmented_img = cv2.threshold(blurred_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Doesn't work well
blurred_layer = viewer.add_image(segmented_img) # Add to napari


# plt.imshow(segmented, cmap='gray')

# =============================================================================
# Perform blob detection
# =============================================================================
# %% Try blob detection

blobs = blob_log(gray.astype(float),max_sigma=35,  min_sigma=30 ,num_sigma=30,threshold=.05,overlap=0.7)
color_blobs = (0,0,255)
width = 1
font = cv2.FONT_HERSHEY_SIMPLEX
with_labels = True
copied_img = np.copy(img)
try:
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    n_label = 0
    for blob in blobs:
        y, x, r = blob
        cv2.circle(copied_img, (int(x),int(y)), int(r),color_blobs , width)
        if with_labels:
            cv2.putText(copied_img,str(n_label),(int(x),int(y)),font,1,color_blobs,width,cv2.LINE_AA)
        n_label += 1
except:
    print(len(blobs))
print(len(blobs))
viewer.add_image(copied_img)
plt.imshow(copied_img, cmap='gray')

# %%
# =============================================================================
# %% Watershed algorithm
# =============================================================================






# %% Run in case of specific image
from skimage.io import imread
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            images.append(imread(img_path))
    return images

imgs = load_images_from_folder(r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\drg_cells')


# %% 
cv2.imshow('image', cl_img_color)

# %%
# filepath = join(dirpath, filename)
filepath = r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_images\images\Mar19bS1C1R2_VLPAGr_200x_y.png'
img = imread(filepath)

## Preprocessing

sigma = 3
img_smooth = ndi.gaussian_filter(img, sigma)


## Adaptive Thresholding
i = 31
SE = (np.mgrid[:i,:i][0] - np.floor(i/2))**2 + (np.mgrid[:i,:i][1] - np.floor(i/2))**2 <= np.floor(i/2)**2
from skimage.filters import rank 
bg = rank.mean(img_smooth, footprint=SE)

mem = img_smooth > bg


## Improving Masks with Binary Morphology

mem_holefilled = ~ndi.binary_fill_holes(~mem) # Short form

i = 15
SE = (np.mgrid[:i,:i][0] - np.floor(i/2))**2 + (np.mgrid[:i,:i][1] - np.floor(i/2))**2 <= np.floor(i/2)**2

pad_size = i+1
mem_padded = np.pad(mem_holefilled, pad_size, mode='reflect')
mem_final = ndi.binary_closing(mem_padded, structure=SE)
mem_final = mem_final[pad_size:-pad_size, pad_size:-pad_size]


## Cell Segmentation by Seeding & Expansion

### Seeding by Distance Transform

dist_trans = ndi.distance_transform_edt(~mem_final)
dist_trans_smooth = ndi.gaussian_filter(dist_trans, sigma=5)

from skimage.feature import peak_local_max
seed_coords = peak_local_max(dist_trans_smooth, min_distance=10)
seeds = np.zeros_like(dist_trans_smooth, dtype=bool)
seeds[tuple(seed_coords.T)] = True

seeds_labeled = ndi.label(seeds)[0]

### Expansion by Watershed

from skimage.segmentation import watershed
ws = watershed(img_smooth, seeds_labeled)


## Postprocessing: Removing Cells at the Image Border

border_mask = np.zeros(ws.shape, dtype=bool)
border_mask = ndi.binary_dilation(border_mask, border_value=1)

clean_ws = np.copy(ws)

for cell_ID in np.unique(ws):
    cell_mask = ws==cell_ID
    cell_border_overlap = np.logical_and(cell_mask, border_mask)
    total_overlap_pixels = np.sum(cell_border_overlap)
    if total_overlap_pixels > 0: 
        clean_ws[cell_mask] = 0

for new_ID, cell_ID in enumerate(np.unique(clean_ws)[1:]): 
    clean_ws[clean_ws==cell_ID] = new_ID+1


## Identifying Cell Edges

edges = np.zeros_like(clean_ws)

for cell_ID in np.unique(clean_ws)[1:]:
    cell_mask = clean_ws==cell_ID
    eroded_cell_mask = ndi.binary_erosion(cell_mask, iterations=1)
    edge_mask = np.logical_xor(cell_mask, eroded_cell_mask)
    edges[edge_mask] = cell_ID


## Extracting Quantitative Measurements

results = {"cell_id"      : [],
            "int_mean"     : [],
            "int_mem_mean" : [],
            "cell_area"    : [],
            "cell_edge"    : []}

for cell_id in np.unique(clean_ws)[1:]:
    cell_mask = clean_ws==cell_id
    edge_mask = edges==cell_id
    results["cell_id"].append(cell_id)
    results["int_mean"].append(np.mean(img[cell_mask]))
    results["int_mem_mean"].append(np.mean(img[edge_mask]))
    results["cell_area"].append(np.sum(cell_mask))
    results["cell_edge"].append(np.sum(edge_mask))

# %%
