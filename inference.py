from glob import glob
import shutil
import torch
from time import strftime
import os
import sys
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

class BaseModel:
    def __init__(self, paths, device, model_class):
        self.model = model_class(paths, device)

    def generate(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the 'generate' method.")


class PreprocessModel(BaseModel):
    def __init__(self, paths, device):
        super().__init__(paths, device, CropAndExtract)

    def generate(self, pic_path, save_dir, preprocess_flag, source_image_flag, pic_size):
        return self.model.generate(pic_path, save_dir, preprocess_flag,
                                   source_image_flag=source_image_flag, pic_size=pic_size)

    def preprocess_source_image(self, pic_path, first_frame_dir, preprocess_flag, pic_size):
        return self.generate(pic_path, first_frame_dir, preprocess_flag,
                                         source_image_flag=True, pic_size=pic_size)

    def preprocess_reference_video(self, ref_video_path, save_dir, preprocess_flag):
        ref_coeff_path, _, _ = self.generate(ref_video_path, save_dir, preprocess_flag,
                                                         source_image_flag=False)
        return ref_coeff_path

    def process_eyeblink_and_pose(self, save_dir, args):
        ref_eyeblink_coeff_path, ref_pose_coeff_path = None, None

        if args.ref_eyeblink is not None:
            ref_eyeblink_coeff_path = self.preprocess_reference_video(self, args.ref_eyeblink, save_dir,
                                                                 args.preprocess)

        if args.ref_pose is not None:
            if args.ref_pose == args.ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_coeff_path = self.preprocess_reference_video(self, args.ref_pose, save_dir,
                                                                 args.preprocess)

        return ref_eyeblink_coeff_path, ref_pose_coeff_path


class AudioToCoeffModel(BaseModel):
    def __init__(self, paths, device):
        super().__init__(paths, device, Audio2Coeff)

    def generate(self, batch, save_dir, pose_style, ref_pose_coeff_path):
        return self.model.generate(batch, save_dir, pose_style, ref_pose_coeff_path)


class AnimateFromCoeffModel(BaseModel):
    def __init__(self, paths, device):
        super().__init__(paths, device, AnimateFromCoeff)

    def generate(self, data, save_dir, pic_path, crop_info, enhancer, background_enhancer, preprocess, img_size):
        return self.model.generate(data, save_dir, pic_path, crop_info,
                                    enhancer=enhancer, background_enhancer=background_enhancer,
                                    preprocess=preprocess, img_size=img_size)


def initialize_paths(args):
    current_root_path = os.path.split(sys.argv[0])[0]
    return init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)


def create_first_frame_dir(save_dir):
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    return first_frame_dir




def main(args):
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

    sadtalker_paths = initialize_paths(args)

    preprocess_model = PreprocessModel(sadtalker_paths, device)
    audio_to_coeff = AudioToCoeffModel(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeffModel(sadtalker_paths, device)

    first_frame_dir = create_first_frame_dir(save_dir)

    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.preprocess_source_image(pic_path, first_frame_dir, args.preprocess, pic_size=args.size)

    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    ref_eyeblink_coeff_path, ref_pose_coeff_path = preprocess_model.process_eyeblink_and_pose(save_dir,args)

    # audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

    # coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                               batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                               expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)

    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info,
                                         enhancer=args.enhancer, background_enhancer=args.background_enhancer,
                                         preprocess=args.preprocess, img_size=args.size)

    shutil.move(result, save_dir + '.mp4')
    print('The generated video is named:', save_dir + '.mp4')

    if not args.verbose:
        shutil.rmtree(save_dir)

    
if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=1,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='full', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" )
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 


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

