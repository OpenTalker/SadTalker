import torch
import yaml
import os

import safetensors
from safetensors.torch import save_file
from yacs.config import CfgNode as CN
import sys

sys.path.append('/apdcephfs/private_shadowcun/SadTalker')

from src.face3d.models import networks

from src.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from src.facerender.modules.mapping import MappingNet
from src.facerender.modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator

from src.audio2pose_models.audio2pose import Audio2Pose
from src.audio2exp_models.networks import SimpleWrapperV2 
from src.test_audio2coeff import load_cpk

size = 256
############ face vid2vid
config_path = os.path.join('src', 'config', 'facerender.yaml')
current_root_path = '.'

path_of_net_recon_model = os.path.join(current_root_path, 'checkpoints', 'epoch_20.pth')
net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='')
checkpoint = torch.load(path_of_net_recon_model, map_location='cpu')    
net_recon.load_state_dict(checkpoint['net_recon'])

with open(config_path) as f:
    config = yaml.safe_load(f)

generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
kp_extractor = KPDetector(**config['model_params']['kp_detector_params'],
                            **config['model_params']['common_params'])
he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                        **config['model_params']['common_params'])
mapping = MappingNet(**config['model_params']['mapping_params'])

def load_cpk_facevid2vid(checkpoint_path, generator=None, discriminator=None, 
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
            print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
    if optimizer_generator is not None:
        optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
    if optimizer_discriminator is not None:
        try:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
        except RuntimeError as e:
            print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
    if optimizer_kp_detector is not None:
        optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
    if optimizer_he_estimator is not None:
        optimizer_he_estimator.load_state_dict(checkpoint['optimizer_he_estimator'])

    return checkpoint['epoch']


def load_cpk_facevid2vid_safetensor(checkpoint_path, generator=None, 
                        kp_detector=None, he_estimator=None,  
                        device="cpu"):

    checkpoint = safetensors.torch.load_file(checkpoint_path)

    if generator is not None:
        x_generator = {}
        for k,v in checkpoint.items():
            if 'generator' in k:
                x_generator[k.replace('generator.', '')] = v
        generator.load_state_dict(x_generator)
    if kp_detector is not None:
        x_generator = {}
        for k,v in checkpoint.items():
            if 'kp_extractor' in k:
                x_generator[k.replace('kp_extractor.', '')] = v
        kp_detector.load_state_dict(x_generator)
    if he_estimator is not None:
        x_generator = {}
        for k,v in checkpoint.items():
            if 'he_estimator' in k:
                x_generator[k.replace('he_estimator.', '')] = v
        he_estimator.load_state_dict(x_generator)
    
    return None

free_view_checkpoint = '/apdcephfs/private_shadowcun/SadTalker/checkpoints/facevid2vid_'+str(size)+'-model.pth.tar'
load_cpk_facevid2vid(free_view_checkpoint, kp_detector=kp_extractor, generator=generator, he_estimator=he_estimator)

wav2lip_checkpoint = os.path.join(current_root_path, 'checkpoints', 'wav2lip.pth')

audio2pose_checkpoint = os.path.join(current_root_path, 'checkpoints', 'auido2pose_00140-model.pth')
audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')

audio2exp_checkpoint = os.path.join(current_root_path, 'checkpoints', 'auido2exp_00300-model.pth')
audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

fcfg_pose = open(audio2pose_yaml_path)
cfg_pose = CN.load_cfg(fcfg_pose)
cfg_pose.freeze()
audio2pose_model = Audio2Pose(cfg_pose, wav2lip_checkpoint)
audio2pose_model.eval()
load_cpk(audio2pose_checkpoint, model=audio2pose_model, device='cpu')

# load audio2exp_model
netG = SimpleWrapperV2()
netG.eval()
load_cpk(audio2exp_checkpoint, model=netG, device='cpu')

class SadTalker(torch.nn.Module):
    def __init__(self, kp_extractor, generator, netG, audio2pose, face_3drecon):
        super(SadTalker, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.audio2exp = netG
        self.audio2pose = audio2pose
        self.face_3drecon = face_3drecon


model = SadTalker(kp_extractor, generator, netG, audio2pose_model, net_recon)

# here, we want to convert it to safetensor
save_file(model.state_dict(), "checkpoints/SadTalker_V0.0.2_"+str(size)+".safetensors")

### test
load_cpk_facevid2vid_safetensor('checkpoints/SadTalker_V0.0.2_'+str(size)+'.safetensors', kp_detector=kp_extractor, generator=generator, he_estimator=None)