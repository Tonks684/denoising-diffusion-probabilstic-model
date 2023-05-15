import os
import numpy as np
import torch
import math 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from einops import rearrange
def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def normalize_16bit_image_to_zero_to_one(img):
    return img.point(lambda p: p*(1/65535.0))

def unnormalize_tensor_to_img(image_tensor, imtype=np.uint16,normalise=True, stack_predictions=False):
    """
    image_tensor: input tensor of size b,c,h,w or b,t,c,h,w 
    returns list of normalised h,w,c images
    """

    if len(image_tensor.size()) == 5:
        # bs, T, channel, width,height
        # 128,1000,1,256,256 
        final_images = []  
        for batch in range(image_tensor.size()[0]):
            for t in range(image_tensor.size()[1]):
                image_numpy =  image_tensor[batch,t,:,:]
                image_numpy = image_numpy.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy,(1,2,0)) 
                image_numpy = image_numpy * 65535.0
                final_images.append(image_numpy.astype(imtype))
    
    if len(image_tensor.size()) == 4:
        # bs,channel, width,height
        # 128,1,256,256 
        final_images = []  
        for batch in range(image_tensor.size()[0]):
                image_numpy =  image_tensor[batch,:,:,:]
                image_numpy = image_numpy.cpu().float().numpy()
                image_numpy = image_numpy.transpose() 
                image_numpy = image_numpy * 65535.0
                final_images.append(image_numpy.astype(imtype))
    # if isinstance(image_tensor, list):
    #     image_numpy = []
    #     for i in range(len(image_tensor)):
    #         image_numpy.append(
    #             unnormalize_tensor_to_img(image_tensor[i])
    #         )
    
    if len(image_tensor.size()) == 3:
        # channel, width,height
        # 1,256,256 
        final_images = []  
        image_numpy =  image_tensor
        image_numpy = image_numpy.cpu().float().numpy()
        image_numpy = image_numpy.transpose() 
        image_numpy = image_numpy * 65535.0
        final_images.append(image_numpy.astype(imtype))

    return final_images

def print_params(net):
    if isinstance(net,list):
        net = net[0]
    num_params = 0
    for params in net.parameters():
        num_params += params.numel()
    print(f'Total Number of params {num_params}')
    return num_params


def sample_grid(samples):
    f, ax = plt.subplots(figsize=(10,4))
    grid_img = make_grid(samples,padding=True, pad_value=1)
    return torchvision.transforms.ToPILImage()(grid_img)

   
def diffusion_proccess(time_steps,noisey_samples):
    """
    time_steps: fixed number of time steps chosen (eg. 250)
    noisey_samples: list of images from 250,259...1 for all batches. BS 3 = len(noisey_samples(750))
    """
    # Extract diffusion per image
    noisey_sample_per_img = [
        list(noisey_samples[i:i+len(time_steps)]) 
        for i in range(0,len(noisey_samples),len(time_steps))]
    # Extract subset of timesteps
    if len(noisey_sample_per_img) <= 10:
        continue
    else:
        subset_timesteps = len(noisey_sample_per_img) // 25
        noisey_sample_per_img = [ noisey_sample_per_img[i+subset_timesteps] 
        for i in range(0,noisey_sample_per_img,subset_timesteps)]

    fig, ax = plt.subplots(len(noisey_sample_per_img),len(noisey_sample_per_img[0]),figsize=(10,5))    # Extract diffusion between batch elements
    for row,j in enumerate(noisey_sample_per_img):
        for col, (time_step, img) in enumerate(zip(time_steps,j)):
            ax[row,col].imshow(img,cmap='gray')
            ax[row,col].set_title(f't={time_step}',fontsize=8)
            ax[row,col].axis('off')
            ax[row,col].grid(False)
    plt.suptitle("Diffusion Process", y=0.9)
    plt.axis('off')
    return fig

