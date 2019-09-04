import torchvision
from PIL import Image, ImageOps
import numpy as np
import torch


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):

        return [self.worker(img) for img in img_group]

class Augmentation(object):

    def __init__(self, flip = True):
        self.flip = True
    def __call__(self, img_group):
        a = np.random.randint(0,4)
        #print(a)
        if a == 0:
            return [img_group[0]]
        elif a == 1:
            verti = img_group[0].transpose(Image.FLIP_LEFT_RIGHT)
            return [verti]
        elif a == 2:
            hori = img_group[0].transpose(Image.FLIP_TOP_BOTTOM)
            return [hori]
        elif a == 3:
            verti = img_group[0].transpose(Image.FLIP_LEFT_RIGHT)
            vertih = verti.transpose(Image.FLIP_TOP_BOTTOM)
            return [vertih]
        #return [img_group[0], verti, hori, vertih]


class Stack(object):

    def __init__(self, roll=False, aug=True):
        self.roll = roll
        self.aug = aug

    def __call__(self, img_group):
        if self.aug:
            #print(type(np.array(img_group[0])))
            #print(var.shape())
            #a = np.concatenate([np.expand_dims(np.expand_dims(np.array(img), 2), 3)for img in img_group], axis=3)
            a = np.expand_dims(np.array(img_group[0]), 2)

        else:
            a = np.expand_dims(np.array(img_group[0]), 2)

        return a



class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True, aug=True):
        self.div = div
        self.aug = aug

    def __call__(self, pic):
        if self.aug:
            # handle numpy array
            img = torch.from_numpy(pic).permute(2,0,1).contiguous()
        else:
            img = torch.from_numpy(pic).permute(2,0,1).contiguous()
        return img
        # return img.float().div(255) if self.div else img.float()
