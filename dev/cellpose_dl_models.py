# %% Experiment with cellpose models
import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from cellpose import utils, io, models, dynamics
from glob import glob
from natsort import natsorted
from pathlib import Path
import random
import logging
logger = logging.getLogger(__name__)

# %% Functions
def run_cellpose_model(
                       img,
                       filename = 'test.tif',
                       do_3D = False,
                       anisotropy = 2.7,
                       model = 'cyto',
                       pretrained_model = None,
                       cellprob_th = 0.0,
                       diameter = 35,
                       channels=[0,0],
                       use_GPU = False
                      ):
    logger = io.logger_setup()
    # Run cellpose model inference
    if pretrained_model:
        model = models.CellposeModel(gpu=use_GPU, pretrained_model=pretrained_model)
    else:
        model = models.CellposeModel(gpu=use_GPU, model_type=model)
    mask, flows, styles = model.eval(img, 
                                            channels=channels, 
                                            diameter=diameter, 
                                            anisotropy=anisotropy, 
                                            do_3D=do_3D, 
                                            #net_avg=False, 
                                            #augment=False, 
                                            cellprob_threshold=cellprob_th)
    # save results so you can load in gui
    io.masks_flows_to_seg(img, mask, flows, model.diam_labels, filename, channels)
    # save results as mask
    io.save_masks(img, mask, flows, filename, png=False, tif=True)
    return mask, flows, styles

def run_cellpose_batch(images, 
        do_3D = True, anisotropy = 2.7,  model = 'cyto2',
        pretrained_model = None, cellprob_th = 0.0, diameter = 35, 
        channels=[0,0], use_GPU = True):
    logger = io.logger_setup()
    if pretrained_model:
        model = models.CellposeModel(gpu=use_GPU, pretrained_model=pretrained_model)
    else:
        model = models.CellposeModel(gpu=use_GPU, model_type=model)
    for img in images:
        run_cellpose_model(img)
  
def train(train_dir, use_GPU, n_epochs, min_train_masks=1, learning_rate=0.1,
          weight_decay=0.0001, pretrained_model=False, model_type=None, test_dir=None, model_name=None, save_path=None,
          nimg_per_epoch=None):
    """
    Wrapper for cellpose model.train(), I've pruned this a bit to only relevent params for this paper/experiment.
    :param train_dir: str
                what directory are training images and *.npy in



    :param use_GPU: bool

    :param n_epochs: int (default, 500)
                how many times to go through whole training set during training. From model.train()

    :param min_train_masks: int (default, 1)
                minimum number of masks an image must have to use in training set. From model.train()

    :param learning_rate: float or list/np.ndarray (default, 0.2)
                learning rate for training, if list, must be same length as n_epochs. From model.train()

    :param weight_decay: float (default, 0.00001)
                From model.train()

    :param pretrained_model: str or list of strings (optional, default False)
        full path to pretrained cellpose model(s), if None or False, no model loaded

    :param model_type: str (optional, default None)
        any model that is available in the GUI, use name in GUI e.g. 'livecell'
        (can be user-trained or model zoo)

    :param test_dir: str (default, None)
                What directory are testing images and *.npy in

    :param model_name: str (default, None)
                name of network, otherwise saved with name as params + training start time. From model.train()

    :param save_path: string (default, None)
            where to save trained model. If None, will be placed in models/train_dir/*

    :param nimg_per_epoch: int (optional, default None)
            minimum number of images to train on per epoch,
            with a small training set (< 8 images) it may help to set to 8. From to model.train()

    :return:
    """
    channels = [0, 0]
    if save_path is None:
        save_path = Path('data/model/' + train_dir.name)
    # check params
    run_str = f'python -m cellpose --use_gpu --verbose --train --dir {train_dir} --pretrained_model {pretrained_model} --model_type {model_type} --chan {channels[0]} --chan2 {channels[1]} --n_epochs {n_epochs} --learning_rate {learning_rate} --weight_decay {weight_decay}'
    if test_dir is not None:
        run_str += f' --test_dir {test_dir}'
    run_str += ' --mask_filter _seg.npy'
    print(run_str)

    # actually start training

    # DEFINE CELLPOSE MODEL (without size model), probably put this to a separate function, to be able to access from main
    if model_type and pretrained_model:
        model = models.CellposeModel(gpu=use_GPU, model_type=model_type, pretrained_model=pretrained_model)
        logger.info(f'model type {model_type} and pretrained model {pretrained_model}')
    if model_type and not pretrained_model:
        model = models.CellposeModel(gpu=use_GPU, model_type=model_type)
        logger.info(f'model type {model_type} only')
    if pretrained_model and not model_type:
        model = models.CellposeModel(gpu=use_GPU, pretrained_model=pretrained_model)
        logger.info(f'pretrained model {pretrained_model} only')

    # get files
    output = io.load_train_test_data(str(train_dir), str(test_dir), mask_filter='_seg.npy')
    train_data, train_labels, _, test_data, test_labels, _ = output
    model.train(train_data, train_labels,
                test_data=test_data,
                test_labels=test_labels,
                channels=channels,
                save_path=save_path,
                save_every=10,
                save_each=True,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                nimg_per_epoch=nimg_per_epoch,
                SGD=True,
                min_train_masks=min_train_masks,
                model_name=model_name)
    return model

def get_masks(directory, use_GPU):
    output = io.load_train_test_data(str(directory), mask_filter='_seg.npy')
    model = models.CellposeModel(gpu=use_GPU)
    train_data, train_labels, train_paths, test_data, test_labels, test_paths = output
    train_flows = dynamics.labels_to_flows(train_labels, files=None, use_gpu=model.gpu, device=model.device)
    nmasks = np.array([label[0].max() for label in train_flows])
    return list(zip(train_paths, nmasks))

# %% Define the paths to the images and masks
# TODO: Use path to transfer to relative
train_files = natsorted(glob(r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_images\images\*.png'))
train_seg = natsorted(glob(r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_masks\masks\*.npy')) 

# %% Load all iamges
from skimage.io import imread
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            images.append(imread(img_path))
    return images

def load_training_data_from_folder(folder_path):
  pass

imgs = load_images_from_folder(r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_images\images')
masks = load_images_from_folder(r'C:\Users\micha\Documents\Projects\Automatic cell body extraction\dev\Flurocells\all_masks\masks')

# %% Run an example of th emodel
mask, flows, styles = run_cellpose_model(imgs[0])

# %% Retrain the model on the new dataset
model = train(train_dir=t_3, model_type="CPx", use_GPU=True, n_epochs=1000, test_dir=test_set_full_dir)
model_path = io_wrapper.get_model_path(log_handler.logs)
io_wrapper.plot_training_stats(io_wrapper.get_training_stats(log_handler.logs), model_name=model_path.name)





# %% Browse result in napari - Start Napari viewer and add images
import napari
anisotropy = 2.7
viewer = napari.Viewer()
for i, img in enumerate(imgs):
    viewer.add_image(img, name=f'Image_{i}')
    # image_layer = viewer.add_image(img, scale = (anisotropy, 1, 1))
    image_layer = viewer.add_image(img)
    label_layer = viewer.add_labels(mask, scale = (anisotropy, 1, 1))
    viewer.add_image(flows[0], scale = (anisotropy, 1, 1), name = 'Flows', visible=False)
# viewer.add_image(img_rescaled)





# %% Run and store all the images, then display a random one
# TODO: Improve and make more efficient
imgs=[] #store all images
for f in train_files: #Read images
  im=io.imread(f)
  n_dim=len(im.shape) #shape of image
  dim=im.shape #dimensions of image
  channel=min(dim) #channel will be dimension with min value usually
  channel_position=dim.index(channel)
  #if no of dim is 3 and channel is first index, swap channel to last index
  if n_dim==3 and channel_position==0: 
    im=im.transpose(1,2,0) # TODO: Why?
    dim=im.shape
  imgs.append(im)

nimg = len(imgs)
print("No of images loaded are: ", nimg)
print("Example Image:")

random_idx = random.choice(range(len(imgs)))
x=imgs[random_idx]
n_dim=len(x.shape)
file_name=os.path.basename(train_files[random_idx])
print(file_name+" has "+str(n_dim)+" dimensions/s")
if n_dim==3:
  channel_image=x.shape[2]
  fig, axs = plt.subplots(1, channel_image,figsize=(12,5))
  print("Image: %s" %(file_name))
  for channel in range(channel_image):
      axs[channel].imshow(x[:,:,channel])
      axs[channel].set_title('Channel '+str(channel+1),size=5)
      axs[channel].axis('off')
  fig.tight_layout()
elif n_dim==2:
  print("One Channel")
  plt.imshow(x)
else:
  print("Channel number invalid or dimensions wrong. Image shape is: "+str(x.shape))


# %% TODO: Train the model on a new dataset

# %% Load a pretrained model
model = models.Cellpose(gpu=False, model_type='cyto')

# %% Test the model on a single image
channels=[0,0] # This means we are processing single-channel greyscale images.
sample_image = imgs[1] # TODO: Switch to random number selection
diameter = 30
flow_threshold = 0.4
cellprob_threshold = 0.0
masks, flows, styles, diams = model.eval(sample_image, diameter=diameter, 
                                         flow_threshold=flow_threshold,cellprob_threshold=cellprob_threshold, 
                                         channels=channels)
# %% Test results
# DISPLAY RESULTS
from cellpose import plot
flowi = flows[0]
fig = plt.figure(figsize=(24,8))
plot.show_segmentation(fig, sample_image, masks, flowi, channels=channels)
plt.tight_layout()
plt.show()

# %% Save the model data
from cellpose import io
#save output of model eval to be loaded in GUI
io.masks_flows_to_seg(imgs, masks, flows, diams, files, channels)



# %% TODO: Load saved data with napari - https://github.com/royerlab/ultrack-i2k2023/blob/main/multiple-labels.ipynb

# Check the image in the viewer
viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_image(img_rescaled)

# FIXME: https://github.com/jluethi/cellpose-hackaton-liberalilab/blob/main/Cellpose_3D_training_workflow.ipynb continue implementing



# # %% Load a cellpose pretrained model
# model = models.Cellpose(gpu=False, model_type='nuclei')

# # Run the model and evaluate the image
# channels = [0,0] # This means we are processing single-channel greyscale images.
# masks, flows, styles, diams = model.eval(image, diameter=None, channels=channels)
# labels = masks.astype(np.uint32)
# stackview.insight(labels)


# # %% Review evaluation
# fig = plt.figure(figsize=(12,5))
# plot.show_segmentation(fig, image, masks, flows[0], channels=channels)
# plt.tight_layout()
# plt.show()
# %%
