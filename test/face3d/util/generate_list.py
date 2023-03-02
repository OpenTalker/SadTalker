"""This script is to generate training list files for Deep3DFaceRecon_pytorch
"""

import os

# save path to training data
def write_list(lms_list, imgs_list, msks_list, mode='train',save_folder='datalist', save_name=''):
    save_path = os.path.join(save_folder, mode)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, save_name + 'landmarks.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in lms_list])

    with open(os.path.join(save_path, save_name + 'images.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in imgs_list])

    with open(os.path.join(save_path, save_name + 'masks.txt'), 'w') as fd:
        fd.writelines([i + '\n' for i in msks_list])   

# check if the path is valid
def check_list(rlms_list, rimgs_list, rmsks_list):
    lms_list, imgs_list, msks_list = [], [], []
    for i in range(len(rlms_list)):
        flag = 'false'
        lm_path = rlms_list[i]
        im_path = rimgs_list[i]
        msk_path = rmsks_list[i]
        if os.path.isfile(lm_path) and os.path.isfile(im_path) and os.path.isfile(msk_path):
            flag = 'true'
            lms_list.append(rlms_list[i])
            imgs_list.append(rimgs_list[i])
            msks_list.append(rmsks_list[i])
        print(i, rlms_list[i], flag)
    return lms_list, imgs_list, msks_list
