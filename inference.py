import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data

def main(args):
    #torch.backends.cudnn.enabled = False

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_code_path = sys.argv[0]
    current_root_path = os.path.split(current_code_path)[0]

    os.environ['TORCH_HOME']=os.path.join(current_root_path, args.checkpoint_dir)

    path_of_lm_croper = os.path.join(current_root_path, args.checkpoint_dir, 'shape_predictor_68_face_landmarks.dat')
    path_of_net_recon_model = os.path.join(current_root_path, args.checkpoint_dir, 'epoch_20.pth')
    dir_of_BFM_fitting = os.path.join(current_root_path, args.checkpoint_dir, 'BFM_Fitting')
    wav2lip_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'wav2lip.pth')

    audio2pose_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'auido2pose_00140-model.pth')
    audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')
    
    audio2exp_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'auido2exp_00300-model.pth')
    audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

    free_view_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'facevid2vid_00189-model.pth.tar')

    if args.preprocess == 'full':
        mapping_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'mapping_00109-model.pth.tar')
        facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender_still.yaml')
    else:
        mapping_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'mapping_00229-model.pth.tar')
        facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender.yaml')

    #init model
    print(path_of_net_recon_model)
    preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)

    print(audio2pose_checkpoint)
    print(audio2exp_checkpoint)
    audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path, 
                                audio2exp_checkpoint, audio2exp_yaml_path, 
                                wav2lip_checkpoint, device)
    
    print(free_view_checkpoint)
    print(mapping_checkpoint)
    animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint, 
                                            facerender_yaml_path, device)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess, source_image_flag=True)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess)
    
    animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess)
    
if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_2.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'resize', 'full'], help="how to preprocess the images" ) 

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)

