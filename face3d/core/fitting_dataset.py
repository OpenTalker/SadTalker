
import numpy as np
from torch.utils import data
import torch
import os
import pickle
import cv2


class FittingDataset(data.Dataset):

    def __init__(self, img_folder, pkl_path, worker_num=1, worker_ind=0):
        super(FittingDataset, self).__init__()
        with open(pkl_path, 'rb') as f:
            lm_dict = pickle.load(f)
        self.lm_dict = lm_dict
        keys = list(lm_dict.keys())
        keys = sorted(keys)
        self.item_list = []
        for k in keys:
            img_full_path = os.path.join(img_folder, str(k)+'.png')
            assert os.path.exists(
                img_full_path), 'file %s does not exist' % img_full_path
            self.item_list.append((k, img_full_path))
        num_insts = len(self.item_list)
        self.start_ind = worker_ind*(num_insts//worker_num)
        if worker_ind == worker_num-1:
            self.group_len = num_insts - self.start_ind
        else:
            self.group_len = num_insts//worker_num

    def __getitem__(self, index):

        k, img_full_path = self.item_list[self.start_ind:self.start_ind +
                                          self.group_len][index]
        lms = self.lm_dict[k]
        img = cv2.imread(img_full_path)[:, :, ::-1].astype(np.float32)
        return torch.tensor(lms), torch.tensor(img), k

    def __len__(self):
        return self.group_len
