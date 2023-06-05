import os
import numpy as np
from PIL import Image
from skimage import io, img_as_float32, transform
import torch
import scipy.io as scio

def get_facerender_data(coeff_path, pic_path, first_coeff_path, audio_path, 
                        batch_size, input_yaw_list=None, input_pitch_list=None, input_roll_list=None, 
                        expression_scale=1.0, still_mode = False, preprocess='crop', size = 256):

    semantic_radius = 13
    video_name = os.path.splitext(os.path.split(coeff_path)[-1])[0]
    txt_path = os.path.splitext(coeff_path)[0]

    data={}

    img1 = Image.open(pic_path)
    source_image = np.array(img1)
    source_image = img_as_float32(source_image)
    source_image = transform.resize(source_image, (size, size, 3))
    source_image = source_image.transpose((2, 0, 1))
    source_image_ts = torch.FloatTensor(source_image).unsqueeze(0)
    source_image_ts = source_image_ts.repeat(batch_size, 1, 1, 1)
    data['source_image'] = source_image_ts
 
    source_semantics_dict = scio.loadmat(first_coeff_path)

    if 'full' not in preprocess.lower():
        source_semantics = source_semantics_dict['coeff_3dmm'][:1,:70]         #1 70
    else:
        source_semantics = source_semantics_dict['coeff_3dmm'][:1,:73]         #1 70

    source_semantics_new = transform_semantic_1(source_semantics, semantic_radius)
    source_semantics_ts = torch.FloatTensor(source_semantics_new).unsqueeze(0)
    source_semantics_ts = source_semantics_ts.repeat(batch_size, 1, 1)
    data['source_semantics'] = source_semantics_ts

    # target 
    generated_dict = scio.loadmat(coeff_path)
    generated_3dmm = generated_dict['coeff_3dmm']
    generated_3dmm[:, :64] = generated_3dmm[:, :64] * expression_scale

    if 'full' in preprocess.lower():
        generated_3dmm = np.concatenate([generated_3dmm, np.repeat(source_semantics[:,70:], generated_3dmm.shape[0], axis=0)], axis=1)

    if still_mode:
        generated_3dmm[:, 64:] = np.repeat(source_semantics[:, 64:], generated_3dmm.shape[0], axis=0)

    with open(txt_path+'.txt', 'w') as f:
        for coeff in generated_3dmm:
            for i in coeff:
                f.write(str(i)[:7]   + '  '+'\t')
            f.write('\n')

    target_semantics_list = [] 
    frame_num = generated_3dmm.shape[0]
    data['frame_num'] = frame_num
    for frame_idx in range(frame_num):
        target_semantics = transform_semantic_target(generated_3dmm, frame_idx, semantic_radius)
        target_semantics_list.append(target_semantics)

    remainder = frame_num%batch_size
    if remainder!=0:
        for _ in range(batch_size-remainder):
            target_semantics_list.append(target_semantics)

    target_semantics_np = np.array(target_semantics_list)             #frame_num 70 semantic_radius*2+1
    target_semantics_np = target_semantics_np.reshape(batch_size, -1, target_semantics_np.shape[-2], target_semantics_np.shape[-1])
    data['target_semantics_list'] = torch.FloatTensor(target_semantics_np)
    data['video_name'] = video_name
    data['audio_path'] = audio_path
    
    if input_yaw_list is not None:
        yaw_c_seq = gen_camera_pose(input_yaw_list, frame_num, batch_size)
        data['yaw_c_seq'] = torch.FloatTensor(yaw_c_seq)
    if input_pitch_list is not None:
        pitch_c_seq = gen_camera_pose(input_pitch_list, frame_num, batch_size)
        data['pitch_c_seq'] = torch.FloatTensor(pitch_c_seq)
    if input_roll_list is not None:
        roll_c_seq = gen_camera_pose(input_roll_list, frame_num, batch_size) 
        data['roll_c_seq'] = torch.FloatTensor(roll_c_seq)
 
    return data

def transform_semantic_1(semantic, semantic_radius):
    semantic_list =  [semantic for i in range(0, semantic_radius*2+1)]
    coeff_3dmm = np.concatenate(semantic_list, 0)
    return coeff_3dmm.transpose(1,0)

def transform_semantic_target(coeff_3dmm, frame_index, semantic_radius):
    num_frames = coeff_3dmm.shape[0]
    seq = list(range(frame_index- semantic_radius, frame_index + semantic_radius+1))
    index = [ min(max(item, 0), num_frames-1) for item in seq ] 
    coeff_3dmm_g = coeff_3dmm[index, :]
    return coeff_3dmm_g.transpose(1,0)

def gen_camera_pose(camera_degree_list, frame_num, batch_size):

    new_degree_list = [] 
    if len(camera_degree_list) == 1:
        for _ in range(frame_num):
            new_degree_list.append(camera_degree_list[0]) 
        remainder = frame_num%batch_size
        if remainder!=0:
            for _ in range(batch_size-remainder):
                new_degree_list.append(new_degree_list[-1])
        new_degree_np = np.array(new_degree_list).reshape(batch_size, -1) 
        return new_degree_np

    degree_sum = 0.
    for i, degree in enumerate(camera_degree_list[1:]):
        degree_sum += abs(degree-camera_degree_list[i])
    
    degree_per_frame = degree_sum/(frame_num-1)
    for i, degree in enumerate(camera_degree_list[1:]):
        degree_last = camera_degree_list[i]
        degree_step = degree_per_frame * abs(degree-degree_last)/(degree-degree_last)
        new_degree_list =  new_degree_list + list(np.arange(degree_last, degree, degree_step))
    if len(new_degree_list) > frame_num:
        new_degree_list = new_degree_list[:frame_num]
    elif len(new_degree_list) < frame_num:
        for _ in range(frame_num-len(new_degree_list)):
            new_degree_list.append(new_degree_list[-1])
    print(len(new_degree_list))
    print(frame_num)

    remainder = frame_num%batch_size
    if remainder!=0:
        for _ in range(batch_size-remainder):
            new_degree_list.append(new_degree_list[-1])
    new_degree_np = np.array(new_degree_list).reshape(batch_size, -1) 
    return new_degree_np
    
