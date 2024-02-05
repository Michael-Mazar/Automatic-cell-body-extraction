# %% Load the library
from utilities import display_img, create_contour_mask, extract_boxes, hough_circle_finder, filter_circles, compute_cell_statistics
import os
import glob
import matplotlib.pyplot as plt
# from skimage import io
from cellpose import io
import cv2
import numpy as np
import csv
from skimage import img_as_float, color, feature
from skimage import measure, segmentation, morphology, filters
from skimage.util import img_as_ubyte
from skimage.feature import blob_dog, blob_log
from skimage.segmentation import inverse_gaussian_gradient
# import mahotas as mh

# %% Load the file
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

# =============================================================================
# Load the images and masks into napari
# =============================================================================
# %% 
# img = train_data[0]

# Alternative loading of iamge
from PIL import Image
# Replace 'your_image.tiff' with the path to your TIFF image file
image_path = r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\drg_cells\2022_10_20_CSU-CY5-50-Pinhole.tif'
# Load the image
img = Image.open(image_path)
img = np.array(img)

# %% Load an array of images
Raw_path = os.path.join(image_dir, '*.tif')
filesRaw = glob.glob(Raw_path)
filesRaw.sort

# %% 
mask = train_labels[0]
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = gray.astype(np.uint16)
img_16bit=gray.copy()
img_8bit = img_as_ubyte(img_16bit)
max_y, max_x = img_8bit.shape
# utilities.display_img(img_8bit)
plt.imshow(img_8bit, cmap='gray')

# =============================================================================
### Truncate intensity histograms at 95th and 15th percentiles ###
# =============================================================================
# %% 
max_pctile = 99.5
max_val = np.percentile(img_16bit.flatten(), max_pctile)
max_limited = img_16bit.copy()
max_limited[img_16bit>max_val] = max_val
threshold_pctile = 15
threshold_d = np.percentile(img_16bit.flatten(), threshold_pctile)
display_img(max_limited, mask)

# =============================================================================
### Image segmentation ###
# =============================================================================
# %%  
img_hist_limited = max_limited.copy()
img_hist_limited[img_hist_limited < threshold_d] = threshold_d
display_img(img_hist_limited, mask)

# =============================================================================
### Inverse gaussian gradiet ###
# =============================================================================
# %% 
temp = img_as_float(img_hist_limited)
gimage = inverse_gaussian_gradient(temp)
display_img(gimage, mask)


# =============================================================================
### Image thresholding ###
# =============================================================================
# %% 
edge_pctile = 5
threshold_g = np.percentile(gimage.flatten(),edge_pctile)
img_thresholded_g = gimage < threshold_g
gimage_8bit_thr=img_as_ubyte(img_thresholded_g)
display_img(gimage_8bit_thr, mask)

# =============================================================================
### Contour search ###
# =============================================================================
# %% 
contours, hierarchy = cv2.findContours(gimage_8bit_thr,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
gimage_8bit_contours = cv2.drawContours(gimage_8bit_thr, contours, -1, (0, 255, 0), 3)
display_img(gimage_8bit_contours,mask)
img_binar = create_contour_mask(img_16bit, contours, pts_threshold = 800)
display_img(img_binar, mask)
img_16bit_cleaned = img_16bit.copy()
img_16bit_cleaned[img_binar == 0] = 0

# =============================================================================
### Blob detection ###
# =============================================================================
# %% 
blobs = blob_log(img_16bit_cleaned.astype(float),max_sigma=35,  min_sigma=30 ,num_sigma=30,threshold=.05,overlap=0.7)
color_blobs = (0,0,255)
width = 1
font = cv2.FONT_HERSHEY_SIMPLEX
with_labels = True

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(img_16bit_cleaned, cmap='plasma')
for cell in blobs:
    y, x, r = cell
    c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
    ax.add_patch(c)
plt.axis(False)

blobs_list=[]
for blob_info in blobs:
    y,x,r = blob_info
    if img_16bit_cleaned[int(y), int(x)] > 0: # this makes sure you only include blobs whose center pixel is on the mask  
        blobs_list.append((y,x))

# =============================================================================
### Clean the thresholds ###
# =============================================================================
# %% # Morphological operations to remove small noise - opening, To remove holes we can use closing
kernel = np.ones((3,3),np.uint8)
cleaning = cv2.morphologyEx(img_16bit_cleaned,cv2.MORPH_OPEN,kernel, iterations = 2)
from skimage.segmentation import clear_border
clean_image = clear_border(cleaning) #Remove edge touching grains
plt.imshow(clean_image, cmap='gray') #This is our image to be segmented further using watershed


# =============================================================================
### Waterthreshold -  Find the boundaries of the cells ###
# =============================================================================
# %%  
ret3, markers = cv2.connectedComponents(clean_image.astype(np.uint8))
plt.imshow(markers)
markers = markers+10
markers = cv2.watershed(img,markers)
#label2rgb - Return an RGB image where color-coded labels are painted over the image.
img2 = color.label2rgb(markers, bg_label=0)
fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img2)
ax[1].imshow(img)

# =============================================================================
### Regional properties of cells ###
# =============================================================================
# %%  
props = measure.regionprops_table(markers, img, 
                          properties=['label',
                                      'area', 'equivalent_diameter',
                                      'mean_intensity', 'solidity', 'orientation',
                                      'perimeter'])
import pandas as pd
df = pd.DataFrame(props)

#To delete small regions...
df = df[df['area'] > 50]
df = df[df['equivalent_diameter'] < 1400]
print(df.head())


# %%
# =============================================================================
### Plot the results ###
# =============================================================================
# Calculate the grid size (rows x columns)
# Calculate the grid size (rows x columns)
num_columns = len(df.columns)
grid_size = int(np.ceil(np.sqrt(num_columns)))

fig, axes = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 4 * grid_size))

# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

# Define a list of colors for the histograms
colors = ['skyblue', 'salmon', 'gold', 'lightgreen', 'plum']

# Iterate through the DataFrame columns and the flattened axes array simultaneously
for i, (col, ax) in enumerate(zip(df.columns, axes_flat)):
    # Plot the histogram with the specified color and additional settings
    df[col].hist(ax=ax, color=colors[i % len(colors)], bins=20, edgecolor='black')
    ax.set_title(f'Histogram of {col}', fontsize=14)
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines only on the y-axis

# Hide any unused axes if the number of plots is less than the number of subplots
for ax in axes_flat[i+1:]:
    ax.axis('off')

plt.tight_layout()
plt.show()

# %%
# =============================================================================
### End ###
# =============================================================================
# %%



#Let us color boundaries in yellow. 
#Remember that watershed assigns boundaries a value of -1
img[markers == -1] = [0,255,255]  

#label2rgb - Return an RGB image where color-coded labels are painted over the image.
img2 = color.label2rgb(markers, bg_label=0)

plt.imshow(img2)
cv2.imshow('Overlay on original image', img)
cv2.imshow('Colored Grains', img2)






# %%
import numpy as np
from scipy import ndimage


# Compute distances from the thresholded image
distances = ndimage.distance_transform_edt(opening_img)
distances = ndimage.gaussian_filter(distances, kernel)

# Find local maxima
maxima = morphology.local_maxima(distances)
maxima = measure.label(maxima, connectivity=2)
surface = distances.max() - distances

# Watershed transformation
markers = measure.label(maxima)
areas = segmentation.watershed(surface, markers, mask=opening_img)

# Label clusters
labeled_clusters, num_clusters = measure.label(opening_img, connectivity=2, return_num=True)

# Join segmentations
joined_labels = segmentation.join_segmentations(areas, labeled_clusters)

# Apply threshold and relabel
labeled_nucl = joined_labels * opening_img
unique_intensities = np.unique(labeled_nucl)

# ret3, markers = cv2.connectedComponents(opening)
# plt.imshow(markers)


# %% 
distances = mh.distance(thresh)
if smooth_distance:
    distance = ndimage.gaussian_filter(distances, kernel)
else:
    distance = distances
maxima = feature.peak_local_max(distance, indices=False, exclude_border=False, min_distance=min_dist)
surface = distance.max() - distance
spots, t = mh.label(maxima) 
areas, lines = mh.cwatershed(surface, spots, return_lines=True)

labeled_clusters, num_clusters= mh.label(thresh, np.ones((3,3), bool))
joined_labels = segmentation.join_segmentations(areas, labeled_clusters)
labeled_nucl = joined_labels * thresh

for index, intensity in enumerate(np.unique(labeled_nucl)):
        labeled_nucl[labeled_nucl == intensity] = index   


# =============================================================================
### End
# =============================================================================
# %%   


# %% 
### Crop images into 50x50 boxes around each neuron center detected with blob_dog and finding edges within each box. ### 
### neuron is true if at least 1 of detected circles lies less than or equal to 5 pixels ###
# bounding_box_dims = [50, 50] # set the height and width of your bounding box
# all_boxes = extract_boxes(img_16bit, blobs_list, bounding_box_dims, max_x, max_y)
# num_circles_to_find = 7
# hough_radii = np.arange(3, 35)
# hough_res = hough_circle_finder(all_boxes, num_circles_to_find, hough_radii, sigma = 4, low_threshold=0, high_threshold=100)
# good_circles, good_box_idx = filter_circles(hough_res, num_circles_to_find, bounding_box_dims, 
#                                 center_deviation_tolerance = 10, near_center_threshold = 3, radius_pct = 95)  
# radii=np.array(good_circles)[:,2]
# good_circles_original_info = np.hstack( (np.array(blobs_list)[good_box_idx,:], radii[:,np.newaxis]) )
# intensities = compute_cell_statistics(img, good_circles_original_info)
# good_circles_coordinates = np.array(blobs_list)[good_box_idx,:]

#%%
### Shows localization of macropinosomes ###

# fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
# ax = axes.ravel()

# ax[0].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
# # ax[1].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')

# # for i in range(len(good_circles_original_info)):
# #     y,x,r = good_circles_original_info[i]
# #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
# #     ax[1].add_patch(c)


# # %% 

# ### Example of cirle fitting in the macropinosome ###
# # utilities.show_box_with_circle(all_boxes, good_box_idx, good_circles, idx2inspect = np.random.randint(len(good_box_idx)))

# final_radii = np.empty([last_frame-first_frame+1, len(radii)])
# final_intensities_YFP= np.empty([last_frame-first_frame+1, len(intensities)])
# final_intensities_FRET=np.empty([last_frame-first_frame+1, len(intensities)])

# final_radii[last_frame-first_frame]=radii
# final_intensities_YFP[last_frame-first_frame]=intensities

# intensities_FRET=utilities.compute_cell_statistics(cell_cropped_fullstack[last_frame,FRET_channel,:,:],good_circles_original_info)
# final_intensities_FRET[last_frame-first_frame]=intensities_FRET


# ### The same analysis is repeated for each time frame ### 

# for stack in reversed(range(first_frame,last_frame)):
#     frame = cell_cropped_fullstack[stack,YFP_channel,:,:]
#     img_16bit=frame.copy()
#     img_8bit = img_as_ubyte(img_16bit)
#     max_y, max_x = img_8bit.shape
#     # utilities.display_img(img_8bit)

#     background_YFP=np.percentile(cell_cropped_fullstack[stack,YFP_channel,:,:].flatten(), background_percent)
#     background_FRET=np.percentile(cell_cropped_fullstack[stack,FRET_channel,:,:].flatten(), background_percent)
#     background_values[stack-first_frame][0]=background_YFP
#     background_values[stack-first_frame][1]=background_FRET

#     max_pctile = 95
#     max_val = np.percentile(img_16bit.flatten(), max_pctile)
#     max_limited = img_16bit.copy()
#     max_limited[img_16bit>max_val] = max_val
#     # utilities.display_img(max_limited)

#     threshold_pctile = 15
#     threshold_d = np.percentile(img_16bit.flatten(), threshold_pctile)

#     img_hist_limited = max_limited.copy()
#     img_hist_limited[img_hist_limited < threshold_d] = threshold_d
#     # utilities.display_img(img_hist_limited)

#     temp = img_as_float(img_hist_limited)
#     gimage = inverse_gaussian_gradient(temp)

#     edge_pctile = 30
#     threshold_g = np.percentile(gimage.flatten(),edge_pctile)
#     img_thresholded_g = gimage < threshold_g
#     gimage_8bit_thr=img_as_ubyte(img_thresholded_g)
#     # utilities.display_img(gimage_8bit_thr)

#     contours, hierarchy = cv2.findContours(gimage_8bit_thr,  
#     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
#     gimage_8bit_contours = cv2.drawContours(gimage_8bit_thr, contours, -1, (0, 255, 0), 3)
#     # utilities.display_img(gimage_8bit_contours)

#     img_binar = utilities.create_contour_mask(img_16bit, contours, pts_threshold = 800)
#     # utilities.display_img(img_binar)

#     img_16bit_cleaned = img_16bit.copy()
#     img_16bit_cleaned[img_binar == 0] = 0

#     blobs = blob_dog(img_16bit, min_sigma = 1, max_sigma=15, threshold=.01)

#     blobs_list=[]
#     for blob_info in blobs:
#         y,x,r = blob_info
#         if img_16bit_cleaned[int(y), int(x)] > 0: # this makes sure you only include blobs whose center pixel is on the mask   
#             blobs_list.append((y,x))

#     # fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
#     # ax = axes.ravel()
#     # ax[0].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
#     # ax[1].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
#     # for filtered_blob in blobs_list:
#     #     y, x = filtered_blob
#     #     c = plt.Circle((x, y), 10, color='red', linewidth=2, fill=False)
#     #     ax[1].add_patch(c)

#     bounding_box_dims = [50, 50] 
#     all_boxes_next = utilities.extract_boxes(img_16bit, blobs_list, bounding_box_dims, max_x, max_y) #boxes based on coordinates of blobs in frame 5
#     num_circles_to_find = 7
#     hough_radii = np.arange(3, 35)
#     hough_res_next = utilities.hough_circle_finder(all_boxes_next, num_circles_to_find, hough_radii, sigma = 4, low_threshold=0, high_threshold=100)
#     good_circles_next, good_box_idx_next = utilities.filter_circles(hough_res_next, num_circles_to_find, bounding_box_dims, 
#                                 center_deviation_tolerance = 5, near_center_threshold = 1, radius_pct = 95)  
#     radii_next=np.array(good_circles_next)[:,2]

#     good_circles_original_info = np.hstack( (np.array(blobs_list)[good_box_idx_next,:], radii_next[:,np.newaxis]) )


#     ### Macropinosome is considered to be the same only if the distance between centers of macropinosomes from consecutive time frames did not exceed 30px ###

#     correct_coordinates = np.zeros([len(good_circles_coordinates), 3])
#     acceptable_distance = 30
#     for circles_old in range(len(good_circles_coordinates)):
#         distance_old=acceptable_distance
#         for circles_new in range(len(good_circles_original_info)):
#             if np.absolute(good_circles_original_info[circles_new,2] - final_radii[stack-(first_frame-1),circles_old]) < 0.3*good_circles_original_info[circles_new,2]:
#                 if np.absolute(good_circles_coordinates[circles_old,0] - good_circles_original_info[circles_new,0]) < acceptable_distance and np.absolute(good_circles_coordinates[circles_old][1] - good_circles_original_info[circles_new][1]) < acceptable_distance:
#                     distance = np.sqrt( (good_circles_coordinates[circles_old][0] - good_circles_original_info[circles_new][0])**2 + (good_circles_coordinates[circles_old][1] - good_circles_original_info[circles_new][1])**2 )
#                     if distance<=distance_old:
#                         distance_old=distance
#                         correct_coordinates[circles_old][0]=good_circles_original_info[circles_new][0]
#                         correct_coordinates[circles_old][1]=good_circles_original_info[circles_new][1]
#                         correct_coordinates[circles_old][2]=good_circles_original_info[circles_new][2]

#     intensities_next_YFP = utilities.compute_cell_statistics(cell_cropped_fullstack[stack,YFP_channel,:,:], correct_coordinates)
#     intensities_next_FRET=utilities.compute_cell_statistics(cell_cropped_fullstack[stack, FRET_channel,:,:], correct_coordinates)
    
#     for i in range(len(correct_coordinates)):
#         good_circles_coordinates[i,0]=correct_coordinates[i,0]
#         good_circles_coordinates[i,1]=correct_coordinates[i,1]
#         if correct_coordinates[i][2]>0:
#             final_intensities_YFP[stack-first_frame,i]=intensities_next_YFP[i]
#             final_intensities_FRET[stack-first_frame,i]=intensities_next_FRET[i]
#             final_radii[stack-first_frame,i]=correct_coordinates[i,2]
#         else:
#             final_intensities_YFP[stack-first_frame,i]=np.nan
#             final_intensities_FRET[stack-first_frame,i]=np.nan
#             final_radii[stack-first_frame,i] = np.nan

# final_radii=final_radii.T
# final_intensities_YFP=final_intensities_YFP.T
# final_intensities_FRET=final_intensities_FRET.T
# background_values=background_values.T

# for i in range (len(final_radii)):
#     if final_radii[i,0] < 7:
#         final_radii[i,:]=np.nan
#         final_intensities_YFP[i,:]=np.nan
#         final_intensities_FRET[i,:]=np.nan

# for i in range (len(final_radii)):
#     counter=0
#     for j in range (len(final_radii[i])):
#         if np.isnan(final_radii[i,j]) == True:
#             counter=counter+1
#     if counter>2:
#         final_radii[i,:]=np.nan
#         final_intensities_YFP[i,:]=np.nan
#         final_intensities_FRET[i,:]=np.nan

# num_mps, num_timesteps = final_radii.shape

# nan_mps = []
# for mp_i in range(num_mps):
#     if np.isnan(final_radii[mp_i,10]):
#         nan_mps.append(mp_i)

# final_radii=np.delete(final_radii,np.array(nan_mps),0)
# final_intensities_YFP=np.delete(final_intensities_YFP,np.array(nan_mps),0)
# final_intensities_FRET=np.delete(final_intensities_FRET,np.array(nan_mps),0)

# YFP_background_column=np.zeros([len(final_radii),(last_frame-first_frame+1)])
# FRET_background_column=np.zeros([len(final_radii),(last_frame-first_frame+1)])

# for i in range(len(final_radii)):
#     YFP_background_column[i]=background_values[0]
#     FRET_background_column[i]=background_values[1]

# row_list=np.concatenate((final_radii,final_intensities_YFP, YFP_background_column, final_intensities_FRET, FRET_background_column),axis=1)
# my_string = tif_file.rsplit('/')[-1]

# column_vector = np.array([my_string]*len(row_list)).reshape(-1,1)

# row_list=np.hstack((column_vector, row_list))

# ### Writing data in csv ###
# with open(csv_name, 'a', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(title_row_list)
#     writer.writerows(row_list)
# # %%
