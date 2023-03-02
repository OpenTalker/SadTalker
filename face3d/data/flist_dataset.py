"""This script defines the custom dataset for Deep3DFaceRecon_pytorch
"""

import os.path
from data.base_dataset import BaseDataset, get_transform, get_affine_mat, apply_img_affine, apply_lm_affine
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import numpy as np
import json
import torch
from scipy.io import loadmat, savemat
import pickle
from util.preprocess import align_img, estimate_norm
from util.load_mats import load_lm3d


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist

def jason_flist_reader(flist):
    with open(flist, 'r') as fp:
        info = json.load(fp)
    return info

def parse_label(label):
    return torch.tensor(np.array(label).astype(np.float32))


class FlistDataset(BaseDataset):
    """
    It requires one directories to host training images '/path/to/data/train'
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        self.lm3d_std = load_lm3d(opt.bfm_folder)
        
        msk_names = default_flist_reader(opt.flist)
        self.msk_paths = [os.path.join(opt.data_root, i) for i in msk_names]

        self.size = len(self.msk_paths) 
        self.opt = opt
        
        self.name = 'train' if opt.isTrain else 'val'
        if '_' in opt.flist:
            self.name += '_' + opt.flist.split(os.sep)[-1].split('_')[0]
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            img (tensor)       -- an image in the input domain
            msk (tensor)       -- its corresponding attention mask
            lm  (tensor)       -- its corresponding 3d landmarks
            im_paths (str)     -- image paths
            aug_flag (bool)    -- a flag used to tell whether its raw or augmented
        """
        msk_path = self.msk_paths[index % self.size]  # make sure index is within then range
        img_path = msk_path.replace('mask/', '')
        lm_path = '.'.join(msk_path.replace('mask', 'landmarks').split('.')[:-1]) + '.txt'

        raw_img = Image.open(img_path).convert('RGB')
        raw_msk = Image.open(msk_path).convert('RGB')
        raw_lm = np.loadtxt(lm_path).astype(np.float32)

        _, img, lm, msk = align_img(raw_img, raw_lm, self.lm3d_std, raw_msk)
        
        aug_flag = self.opt.use_aug and self.opt.isTrain
        if aug_flag:
            img, lm, msk = self._augmentation(img, lm, self.opt, msk)
        
        _, H = img.size
        M = estimate_norm(lm, H)
        transform = get_transform()
        img_tensor = transform(img)
        msk_tensor = transform(msk)[:1, ...]
        lm_tensor = parse_label(lm)
        M_tensor = parse_label(M)


        return {'imgs': img_tensor, 
                'lms': lm_tensor, 
                'msks': msk_tensor, 
                'M': M_tensor,
                'im_paths': img_path, 
                'aug_flag': aug_flag,
                'dataset': self.name}

    def _augmentation(self, img, lm, opt, msk=None):
        affine, affine_inv, flip = get_affine_mat(opt, img.size)
        img = apply_img_affine(img, affine_inv)
        lm = apply_lm_affine(lm, affine, flip, img.size)
        if msk is not None:
            msk = apply_img_affine(msk, affine_inv, method=Image.BILINEAR)
        return img, lm, msk
    



    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.size
