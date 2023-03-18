# check the sync of 3dmm feature and the audio
import  os
import numpy as np

import cv2
import numpy as np
from core import get_recon_model
import os
import torch
import pickle
import subprocess, platform


# draft
def gen_composed_video(device, coeff_ground, audio_path, save_path, exp_dim=64):
    
    tmp_video_path = '/tmp/tmp.mp4'
    v_info_path =  './checkpoints/v_info.pkl'
    original_pkl = './checkpoints/fitting_res.pkl'
    
    with open(original_pkl, 'rb') as f:
        fitting_dict = pickle.load(f)

    id_tensor = torch.tensor(fitting_dict['id'], device=device)
    tex_tensor = torch.tensor(fitting_dict['tex'], device=device)
    
    with open(v_info_path, 'rb') as f:
        video_info = pickle.load(f)

    bbox = video_info['bbox']
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    res_dict = fitting_dict['fitting_res']

    recon_model = get_recon_model(model='bfm09',
                                  device=device,
                                  batch_size=1,
                                  img_size=256)

    video = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc(
        *'mp4v'), video_info['fps'], (video_info['frame_w'], video_info['frame_h']))
    
    for k in range(coeff_ground.shape[0]):
        orig_frame = np.zeros((video_info['frame_h'],video_info['frame_w'],3))
        exp_tensor = coeff_ground[k:(k+1),0:exp_dim]
        cur_exp_tensor = torch.tensor(exp_tensor, device=device).float()
        cur_rot_tensor = torch.tensor(coeff_ground[k:(k+1),exp_dim:exp_dim+3], device=device) # 
        cur_trans_tensor = torch.tensor(coeff_ground[k:(k+1),exp_dim+3:exp_dim+6], device=device)
        cur_gamma_tensor = torch.tensor(res_dict[0]['gamma'], device=device)

        pred_dict = recon_model(recon_model.merge_coeffs(
            id_tensor, cur_exp_tensor, tex_tensor,
            cur_rot_tensor, cur_gamma_tensor, cur_trans_tensor), render=True)

        predicted_landmark = pred_dict['lms_proj']
        predicted_landmark = predicted_landmark.cpu().numpy().squeeze()

        rendered_img = pred_dict['rendered_img']
        rendered_img = rendered_img.cpu().numpy().squeeze()
        out_img = rendered_img[:, :, :3].astype(np.uint8)
        out_mask = (rendered_img[:, :, 3] > 0).astype(np.uint8)
        resized_out_img = cv2.resize(out_img, (face_w, face_h))[:, :, ::-1]
        resized_mask = cv2.resize(
            out_mask, (face_w, face_h), cv2.INTER_NEAREST)[..., None]

        composed_face = orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :] * \
            (1 - resized_mask) + resized_out_img * resized_mask
        orig_frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = composed_face

        video.write(np.uint8(orig_frame))
    video.release()

    command = 'ffmpeg -v quiet -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, tmp_video_path, save_path)
    subprocess.call(command, shell=platform.system() != 'Windows')

