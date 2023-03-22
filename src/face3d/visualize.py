# check the sync of 3dmm feature and the audio
import cv2
import numpy as np
from src.face3d.models.bfm import ParametricFaceModel
from src.face3d.models.facerecon_model import FaceReconModel
import torch
import subprocess, platform
import scipy.io as scio
from tqdm import tqdm 

# draft
def gen_composed_video(args, device, first_frame_coeff, coeff_path, audio_path, save_path, exp_dim=64):
    
    coeff_first = scio.loadmat(first_frame_coeff)['full_3dmm']

    coeff_pred = scio.loadmat(coeff_path)['coeff_3dmm']

    coeff_full = np.repeat(coeff_first, coeff_pred.shape[0], axis=0) # 257

    coeff_full[:, 80:144] = coeff_pred[:, 0:64]
    coeff_full[:, 224:227]  = coeff_pred[:, 64:67] # 3 dim translation
    coeff_full[:, 254:]  = coeff_pred[:, 67:] # 3 dim translation

    tmp_video_path = '/tmp/face3dtmp.mp4'

    facemodel = FaceReconModel(args)
    
    video = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (224, 224))

    for k in tqdm(range(coeff_pred.shape[0]), 'face3d rendering:'):
        cur_coeff_full = torch.tensor(coeff_full[k:k+1], device=device)

        facemodel.forward(cur_coeff_full, device)

        predicted_landmark = facemodel.pred_lm # TODO.
        predicted_landmark = predicted_landmark.cpu().numpy().squeeze()

        rendered_img = facemodel.pred_face
        rendered_img = 255. * rendered_img.cpu().numpy().squeeze().transpose(1,2,0)
        out_img = rendered_img[:, :, :3].astype(np.uint8)

        video.write(np.uint8(out_img[:,:,::-1]))

    video.release()

    command = 'ffmpeg -v quiet -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, tmp_video_path, save_path)
    subprocess.call(command, shell=platform.system() != 'Windows')

