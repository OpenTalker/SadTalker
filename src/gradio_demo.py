import torch, uuid
import os, sys, shutil
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data

from pydub import AudioSegment

def mp3_to_wav(mp3_filename,wav_filename,frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename,format="wav")


class SadTalker():

    def __init__(self, checkpoint_path='checkpoints', config_path='src/config', lazy_load=False):

        if torch.cuda.is_available() :
            device = "cuda"
        else:
            device = "cpu"
        
        self.device = device

        os.environ['TORCH_HOME']= checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path

        self.path_of_lm_croper = os.path.join( checkpoint_path, 'shape_predictor_68_face_landmarks.dat')
        self.path_of_net_recon_model = os.path.join( checkpoint_path, 'epoch_20.pth')
        self.dir_of_BFM_fitting = os.path.join( checkpoint_path, 'BFM_Fitting')
        self.wav2lip_checkpoint = os.path.join( checkpoint_path, 'wav2lip.pth')

        self.audio2pose_checkpoint = os.path.join( checkpoint_path, 'auido2pose_00140-model.pth')
        self.audio2pose_yaml_path = os.path.join( config_path, 'auido2pose.yaml')
    
        self.audio2exp_checkpoint = os.path.join( checkpoint_path, 'auido2exp_00300-model.pth')
        self.audio2exp_yaml_path = os.path.join( config_path, 'auido2exp.yaml')

        self.free_view_checkpoint = os.path.join( checkpoint_path, 'facevid2vid_00189-model.pth.tar')

        self.lazy_load = lazy_load

        if not self.lazy_load:
            #init model
            
            print(self.audio2pose_checkpoint)
            self.audio_to_coeff = Audio2Coeff(self.audio2pose_checkpoint, self.audio2pose_yaml_path, 
                                    self.audio2exp_checkpoint, self.audio2exp_yaml_path, self.wav2lip_checkpoint, self.device)

            print(self.path_of_lm_croper)
            self.preprocess_model = CropAndExtract(self.path_of_lm_croper, self.path_of_net_recon_model, self.dir_of_BFM_fitting, self.device)

    def test(self, source_image, driven_audio, preprocess='crop', still_mode=False, use_enhancer=False, result_dir='./results/'):

        ### crop: only model,

        if self.lazy_load:
            #init model
            
            print(self.audio2pose_checkpoint)
            self.audio_to_coeff = Audio2Coeff(self.audio2pose_checkpoint, self.audio2pose_yaml_path, 
                                    self.audio2exp_checkpoint, self.audio2exp_yaml_path, self.wav2lip_checkpoint, self.device)
        
            print(self.path_of_lm_croper)
            self.preprocess_model = CropAndExtract(self.path_of_lm_croper, self.path_of_net_recon_model, self.dir_of_BFM_fitting, self.device)

        if preprocess == 'full': 
            self.mapping_checkpoint = os.path.join(self.checkpoint_path, 'mapping_00109-model.pth.tar')
            self.facerender_yaml_path = os.path.join(self.config_path, 'facerender_still.yaml')
        else:
            self.mapping_checkpoint = os.path.join(self.checkpoint_path, 'mapping_00229-model.pth.tar')
            self.facerender_yaml_path = os.path.join(self.config_path, 'facerender.yaml')

        print(self.free_view_checkpoint)
        self.animate_from_coeff = AnimateFromCoeff(self.free_view_checkpoint, self.mapping_checkpoint, 
                                            self.facerender_yaml_path, self.device)

        time_tag = str(uuid.uuid4())
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)

        input_dir = os.path.join(save_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        print(source_image)
        pic_path = os.path.join(input_dir, os.path.basename(source_image)) 
        shutil.move(source_image, input_dir)

        if os.path.isfile(driven_audio):
            audio_path = os.path.join(input_dir, os.path.basename(driven_audio))  

            #### mp3 to wav
            if '.mp3' in audio_path:
                mp3_to_wav(driven_audio, audio_path.replace('.mp3', '.wav'), 16000)
                audio_path = audio_path.replace('.mp3', '.wav')
            else:
                shutil.move(driven_audio, input_dir)
        else:
            raise AttributeError("error audio")


        os.makedirs(save_dir, exist_ok=True)
        pose_style = 0
        #crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(pic_path, first_frame_dir, preprocess)
        
        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        #audio2ceoff
        batch = get_data(first_coeff_path, audio_path, self.device, ref_eyeblink_coeff_path=None, still=still_mode) # longer audio?
        coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style)
        #coeff2video
        batch_size = 2
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, still_mode=still_mode, preprocess=preprocess)
        return_path = self.animate_from_coeff.generate(data, save_dir,  pic_path, crop_info, enhancer='gfpgan' if use_enhancer else None, preprocess=preprocess)
        video_name = data['video_name']
        print(f'The generated video is named {video_name} in {save_dir}')

        if self.lazy_load:
            del self.preprocess_model
            del self.audio_to_coeff
            del self.animate_from_coeff

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        import gc; gc.collect()
        
        return return_path

    