import os
import numpy as np
import torch
import math 
import args


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
    # print(image_tensor)
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(
                unnormalize_tensor_to_img(image_tensor[i])
            )
        return image_numpy
    
    if len(image_tensor.size()) == 5:
        # bs, T, channel, width,height
        # 128,1000,1,256,256
        
        for batch in range(image_tensor.size()[0]):
            image_numpy = []
            for t in range(image_tensor.size()[1]):
                image_numpy.append(
                unnormalize_tensor_to_img(image_tensor[batch,t,:,:,:])
            )
        return image_numpy

    if len(image_tensor.size()) == 4:
        # bs,channel, width,height
        # 128,1,256,256
        
        for batch in range(image_tensor.size()[0]):
            image_numpy = []
            for t in range(image_tensor.size()[1]):
                image_numpy.append(
                unnormalize_tensor_to_img(image_tensor[batch,:,:,:])
            )
        return image_numpy


    image_numpy = image_tensor.cpu().float().numpy()
    # if normalise:
        # print(f'pre tranpose shape: {image_numpy.shape}')
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 65535.0
    # if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
    #     image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)
