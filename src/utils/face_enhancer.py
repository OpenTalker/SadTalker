import os
from basicsr.utils import imwrite

from gfpgan import GFPGANer

from tqdm import tqdm

def enhancer(images, method='gfpgan'):

    # ------------------------ set up GFPGAN restorer ------------------------
    if  method == 'gfpgan':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif method == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    elif method == 'codeformer':
        arch = 'CodeFormer'
        channel_multiplier = 2
        model_name = 'CodeFormer'
        url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    else:
        raise ValueError(f'Wrong model version {method}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    
    if not os.path.isfile(model_path):
        model_path = os.path.join('checkpoints', model_name + '.pth')
    
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=None)

    # ------------------------ restore ------------------------
    restored_img = [] 
    for idx in tqdm(range(len(images)), 'Face Enhancer:'):
        
        # restore faces and background if necessary
        cropped_faces, restored_faces, _ = restorer.enhance(
            images[idx],
            has_aligned=True,
            only_center_face=False,
            paste_back=True,
            weight=0.5)
        
        restored_img += restored_faces
       
    return restored_img