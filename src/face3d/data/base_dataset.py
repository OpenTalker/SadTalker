"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        # self.root = opt.dataroot
        self.current_epoch = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_transform(grayscale=False):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)

def get_affine_mat(opt, size):
    shift_x, shift_y, scale, rot_angle, flip = 0., 0., 1., 0., False
    w, h = size

    if 'shift' in opt.preprocess:
        shift_pixs = int(opt.shift_pixs)
        shift_x = random.randint(-shift_pixs, shift_pixs)
        shift_y = random.randint(-shift_pixs, shift_pixs)
    if 'scale' in opt.preprocess:
        scale = 1 + opt.scale_delta * (2 * random.random() - 1)
    if 'rot' in opt.preprocess:
        rot_angle = opt.rot_angle * (2 * random.random() - 1)
        rot_rad = -rot_angle * np.pi/180
    if 'flip' in opt.preprocess:
        flip = random.random() > 0.5

    shift_to_origin = np.array([1, 0, -w//2, 0, 1, -h//2, 0, 0, 1]).reshape([3, 3])
    flip_mat = np.array([-1 if flip else 1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape([3, 3])
    shift_mat = np.array([1, 0, shift_x, 0, 1, shift_y, 0, 0, 1]).reshape([3, 3])
    rot_mat = np.array([np.cos(rot_rad), np.sin(rot_rad), 0, -np.sin(rot_rad), np.cos(rot_rad), 0, 0, 0, 1]).reshape([3, 3])
    scale_mat = np.array([scale, 0, 0, 0, scale, 0, 0, 0, 1]).reshape([3, 3])
    shift_to_center = np.array([1, 0, w//2, 0, 1, h//2, 0, 0, 1]).reshape([3, 3])
    
    affine = shift_to_center @ scale_mat @ rot_mat @ shift_mat @ flip_mat @ shift_to_origin    
    affine_inv = np.linalg.inv(affine)
    return affine, affine_inv, flip

def apply_img_affine(img, affine_inv, method=Image.BICUBIC):
    return img.transform(img.size, Image.AFFINE, data=affine_inv.flatten()[:6], resample=Image.BICUBIC)

def apply_lm_affine(landmark, affine, flip, size):
    _, h = size
    lm = landmark.copy()
    lm[:, 1] = h - 1 - lm[:, 1]
    lm = np.concatenate((lm, np.ones([lm.shape[0], 1])), -1)
    lm = lm @ np.transpose(affine)
    lm[:, :2] = lm[:, :2] / lm[:, 2:]
    lm = lm[:, :2]
    lm[:, 1] = h - 1 - lm[:, 1]
    if flip:
        lm_ = lm.copy()
        lm_[:17] = lm[16::-1]
        lm_[17:22] = lm[26:21:-1]
        lm_[22:27] = lm[21:16:-1]
        lm_[31:36] = lm[35:30:-1]
        lm_[36:40] = lm[45:41:-1]
        lm_[40:42] = lm[47:45:-1]
        lm_[42:46] = lm[39:35:-1]
        lm_[46:48] = lm[41:39:-1]
        lm_[48:55] = lm[54:47:-1]
        lm_[55:60] = lm[59:54:-1]
        lm_[60:65] = lm[64:59:-1]
        lm_[65:68] = lm[67:64:-1]
        lm = lm_
    return lm
