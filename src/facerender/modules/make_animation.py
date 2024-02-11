from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
from PIL import Image
from skimage import io, img_as_float32, transform
import os

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, 1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical['value']    # (bs, k, 3) 
    yaw, pitch, roll= he['yaw'], he['pitch'], he['roll']      
    yaw = headpose_pred_to_degree(yaw) 
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he:
        yaw = he['yaw_in']
    if 'pitch_in' in he:
        pitch = he['pitch_in']
    if 'roll_in' in he:
        roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    if wo_exp:
        exp =  exp*0  
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0]*0
    t[:, 2] = t[:, 2]*0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {'value': kp_transformed}




def extract_eye_region(eye_landmarks, source_np):
    min_x = min(landmark[0] for landmark in eye_landmarks)
    max_x = max(landmark[0] for landmark in eye_landmarks)
    min_y = min(landmark[1] for landmark in eye_landmarks)
    max_y = max(landmark[1] for landmark in eye_landmarks)
    return source_np[int(min_y):int(max_y), int(min_x):int(max_x)]

def paste_eye_region(generated_img, eye_region, top_left_corner):
    y, x = top_left_corner
    h, w, _ = eye_region.shape
    generated_img[int(y):int(y+h), int(x):int(x+w)] = eye_region
    return generated_img


def make_animation(landmarks, save_dir, pic_name, source_semantics, target_semantics,
                            generator, kp_detector, he_estimator, mapping, 
                            yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                            use_exp=True, use_half=False, size=256, device='cpu', restore_eyes=False, only_first_semantic=True, only_first_image=True):
    
    ### original sadtalker performed inference for:
    # 1st frame
    # 1st landmarks
    # for each audio2coeff predicted landmark -> output 1 frame
    ### we change it to iterate frames and landmarks along with each audio2coeff
    # Nth frame
    # Nth landmark
    # Nth audio2coeff predicted landmark -> output Nth frame
    with torch.no_grad():
        predictions = []

        n_total_frames = target_semantics.shape[1]

        # note: can be batched
        for frame_idx in tqdm(range(n_total_frames), 'Face Renderer:'):
            png_path = os.path.join(save_dir, 'first_frame_dir', os.path.basename(pic_name).split('.')[0] + f'-{frame_idx}.png')
            if not os.path.isfile(png_path):
                break
            
            source_image_np = img_as_float32(np.array(Image.open(png_path)))
            source_image = transform.resize(source_image_np, (size, size, 3)).transpose((2, 0, 1))
            source_image = torch.FloatTensor(source_image).unsqueeze(0).to(device) # 1, 3, 256, 256

            if frame_idx == 0 or (not only_first_image):
                source_image_for_inference = source_image

            if frame_idx == 0 or (not only_first_semantic):
                kp_canonical = kp_detector(source_image)

                he_source = mapping(source_semantics[frame_idx].unsqueeze(0)) # each.shape=[1, 45]
                kp_source = keypoint_transformation(kp_canonical, he_source) # 2, 15, 3

            # still check the dimension
            # print(target_semantics.shape, source_semantics.shape)
            target_semantics_frame = target_semantics[frame_idx].unsqueeze(0)
            he_driving = mapping(target_semantics_frame)
            if yaw_c_seq is not None:
                he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
            if pitch_c_seq is not None:
                he_driving['pitch_in'] = pitch_c_seq[:, frame_idx] 
            if roll_c_seq is not None:
                he_driving['roll_in'] = roll_c_seq[:, frame_idx] 
            
            kp_driving = keypoint_transformation(kp_canonical, he_driving)

            kp_norm = kp_driving
            out = generator(source_image_for_inference, kp_source=kp_source, kp_driving=kp_norm)
            '''
            source_image_new = out['prediction'].squeeze(1)
            kp_canonical_new =  kp_detector(source_image_new)
            he_source_new = he_estimator(source_image_new) 
            kp_source_new = keypoint_transformation(kp_canonical_new, he_source_new, wo_exp=True)
            kp_driving_new = keypoint_transformation(kp_canonical_new, he_driving, wo_exp=True)
            out = generator(source_image_new, kp_source=kp_source_new, kp_driving=kp_driving_new)
            '''

            pred_img = out['prediction'].squeeze() # 3, 256, 256
            if not restore_eyes:
                predictions.append(pred_img)
                continue
            
            # else: restore eyes
            pred_img_np = pred_img.cpu().detach().numpy().transpose(1, 2, 0)  
            pred_img_np = (pred_img_np * 255).astype(np.uint8)

            landmarks_for_img = landmarks[frame_idx]
            
            # note: this just does 2 square crops and pastes back
            # this can be made alternative ways for better results, but needs more research,
            # which is out of scope for this assignment. Some things to try out
            # - seameless paste onto the final image
            # - calculating an oval around the eye location and using those to paste
            # - running face landmark detection again on the generated image, and use those
            # in combindation with the original source landmarks to do the paste

            # facexlib landmark indices
            left_eye_indices = [36, 37, 38, 39, 40, 41]
            right_eye_indices = [42, 43, 44, 45, 46, 47]

            left_eye_landmarks = [landmarks_for_img[i] for i in left_eye_indices]
            right_eye_landmarks = [landmarks_for_img[i] for i in right_eye_indices]
            left_eye_region = extract_eye_region(left_eye_landmarks, source_image_np)
            right_eye_region = extract_eye_region(right_eye_landmarks, source_image_np)

            left_eye_top_left = (left_eye_landmarks[0][0], left_eye_landmarks[0][1]) 
            right_eye_top_left = (right_eye_landmarks[0][0], right_eye_landmarks[0][1]) 
            pred_img_with_eyes = paste_eye_region(pred_img_np, left_eye_region, left_eye_top_left)
            pred_img_with_eyes = paste_eye_region(pred_img_with_eyes, right_eye_region, right_eye_top_left)

            final_pred_img = torch.from_numpy(pred_img_with_eyes.transpose(2, 0, 1)) / 255.0
            predictions.append(final_pred_img)
        
        predictions_ts = torch.stack(predictions)
    return predictions_ts

class AnimateModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        
        source_image = x['source_image']
        source_semantics = x['source_semantics']
        target_semantics = x['target_semantics']
        yaw_c_seq = x['yaw_c_seq']
        pitch_c_seq = x['pitch_c_seq']
        roll_c_seq = x['roll_c_seq']

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor,
                                        self.mapping, use_exp = True,
                                        yaw_c_seq=yaw_c_seq, pitch_c_seq=pitch_c_seq, roll_c_seq=roll_c_seq)
        
        return predictions_video