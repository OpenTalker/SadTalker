import os
import cv2
import yaml
import numpy as np
import warnings
from skimage import img_as_ubyte
import safetensors
import safetensors.torch
warnings.filterwarnings('ignore')


import imageio
import torch
import torchvision


from src.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from src.facerender.modules.mapping import MappingNet
from src.facerender.modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from src.facerender.modules.make_animation import make_animation

from pydub import AudioSegment
from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
from src.utils.paste_pic import paste_pic
from src.utils.videoio import save_video_with_watermark


class AnimateFromCoeff:
    """
    Class for animating facial coefficients using specified models.

    Args:
        sadtalker_path (dict): Dictionary containing paths to configuration files and checkpoints.
        device (str): Device to run the models on (e.g., "cpu" or "cuda").

    Attributes:
        generator (OcclusionAwareSPADEGenerator): Generator model for animation.
        kp_extractor (KPDetector): Key-point extractor model.
        he_estimator (HEEstimator): Head pose estimator model.
        mapping (MappingNet): Mapping network model.
        device (str): Specified device for model execution.
    """

    def __init__(self, sadtalker_path, device):
        """
        Initialize the AnimateFromCoeff class.

        Args:
            sadtalker_path (dict): Dictionary containing paths to configuration files and checkpoints.
            device (str): Device to run the models on (e.g., "cpu" or "cuda").

        Attributes:
            generator (OcclusionAwareSPADEGenerator): Generator model for animation.
            kp_extractor (KPDetector): Key-point extractor model.
            he_estimator (HEEstimator): Head pose estimator model.
            mapping (MappingNet): Mapping network model.
            device (str): Specified device for model execution.
        """
        with open(sadtalker_path['facerender_yaml']) as f:
            config = yaml.safe_load(f)['model_params']

        self.generator = OcclusionAwareSPADEGenerator(**config['generator_params'], **config['common_params']).to(
            device).eval()
        self.kp_extractor = KPDetector(**config['kp_detector_params'], **config['common_params']).to(device).eval()
        self.he_estimator = HEEstimator(**config['he_estimator_params'], **config['common_params']).to(
            device).eval()
        self.mapping = MappingNet(**config['mapping_params']).to(device).eval()

        if sadtalker_path is not None:
            if 'checkpoint' in sadtalker_path:
                self.load_cpk_facevid2vid_safetensor(sadtalker_path['checkpoint'], kp_detector=self.kp_extractor,
                                                     generator=self.generator, he_estimator=None)
            else:
                self.load_cpk_facevid2vid(sadtalker_path['free_view_checkpoint'], kp_detector=self.kp_extractor,
                                          generator=self.generator, he_estimator=self.he_estimator)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.")

        if sadtalker_path['mappingnet_checkpoint'] is not None:
            self.load_cpk_mapping(sadtalker_path['mappingnet_checkpoint'], mapping=self.mapping)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.")

        self.kp_extractor.eval()
        self.generator.eval()
        self.he_estimator.eval()
        self.mapping.eval()

        self.device = device

    def load_cpk_facevid2vid_safetensor(self, checkpoint_path, generator=None,
                                        kp_detector=None, he_estimator=None,
                                        device="cpu"):

        checkpoint = safetensors.torch.load_file(checkpoint_path)

        if generator is not None:
            x_generator = {}
            for k, v in checkpoint.items():
                if 'generator' in k:
                    x_generator[k.replace('generator.', '')] = v
            generator.load_state_dict(x_generator)
        if kp_detector is not None:
            x_generator = {}
            for k, v in checkpoint.items():
                if 'kp_extractor' in k:
                    x_generator[k.replace('kp_extractor.', '')] = v
            kp_detector.load_state_dict(x_generator)
        if he_estimator is not None:
            x_generator = {}
            for k, v in checkpoint.items():
                if 'he_estimator' in k:
                    x_generator[k.replace('he_estimator.', '')] = v
            he_estimator.load_state_dict(x_generator)

        return None

    def load_cpk_facevid2vid(self, checkpoint_path, generator=None, discriminator=None,
                             kp_detector=None, he_estimator=None, optimizer_generator=None,
                             optimizer_discriminator=None, optimizer_kp_detector=None,
                             optimizer_he_estimator=None, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if he_estimator is not None:
            he_estimator.load_state_dict(checkpoint['he_estimator'])
        if discriminator is not None:
            try:
                discriminator.load_state_dict(checkpoint['discriminator'])
            except:
                print('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_he_estimator is not None:
            optimizer_he_estimator.load_state_dict(checkpoint['optimizer_he_estimator'])

        return checkpoint['epoch']

    def load_cpk_mapping(self, checkpoint_path, mapping=None, discriminator=None,
                         optimizer_mapping=None, optimizer_discriminator=None, device='cpu'):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        if mapping is not None:
            mapping.load_state_dict(checkpoint['mapping'])
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if optimizer_mapping is not None:
            optimizer_mapping.load_state_dict(checkpoint['optimizer_mapping'])
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])

        return checkpoint['epoch']

    def generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None,
                 preprocess='crop', img_size=256):
        """
        Generate an animated video from given input.

        Args:
            x (dict): Input data dictionary.
            video_save_dir (str): Directory to save the generated video.
            pic_path (str): Path to the picture file.
            crop_info (list): List containing crop information.
            enhancer (str, optional): Type of enhancer to be applied. Defaults to None.
            background_enhancer (str, optional): Background enhancer type. Defaults to None.
            preprocess (str, optional): Preprocessing method. Defaults to 'crop'.
            img_size (int, optional): Size of the output image. Defaults to 256.

        Returns:
            str: Path to the generated video.
        """
        source_image = x['source_image'].type(torch.FloatTensor).to(self.device)
        source_semantics = x['source_semantics'].type(torch.FloatTensor).to(self.device)
        target_semantics = x['target_semantics_list'].type(torch.FloatTensor).to(self.device)

        yaw_c_seq = x['yaw_c_seq'].type(torch.FloatTensor).to(self.device) if 'yaw_c_seq' in x else None
        pitch_c_seq = x['pitch_c_seq'].type(torch.FloatTensor).to(self.device) if 'pitch_c_seq' in x else None
        roll_c_seq = x['roll_c_seq'].type(torch.FloatTensor).to(self.device) if 'roll_c_seq' in x else None

        frame_num = x['frame_num']

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                           self.generator, self.kp_extractor, self.he_estimator, self.mapping,
                                           yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp=True)

        predictions_video = predictions_video.reshape((-1,) + predictions_video.shape[2:])
        predictions_video = predictions_video[:frame_num]

        video = [np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32) for image in predictions_video]
        result = img_as_ubyte(video)

        video_name = x['video_name'] + '.mp4'
        path = os.path.join(video_save_dir, 'temp_' + video_name)

        imageio.mimsave(path, result, fps=float(25))

        audio_path = x['audio_path']
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name + '.wav')
        start_time = 0
        sound = AudioSegment.from_file(audio_path)
        frames = frame_num
        end_time = start_time + frames * 1 / 25 * 1000
        word1 = sound.set_frame_rate(16000)
        word = word1[start_time:end_time]
        word.export(new_audio_path, format="wav")

        video_name_full = x['video_name']  + '_full.mp4'
        full_video_path = os.path.join(video_save_dir, video_name_full)
        return_path = full_video_path
        paste_pic(path, pic_path, crop_info, new_audio_path, full_video_path)
        print(f'The generated video is named {video_save_dir}/{video_name_full}')

        os.remove(path)
        os.remove(new_audio_path)
        return return_path

