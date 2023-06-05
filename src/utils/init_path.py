import os
import glob

def init_path(checkpoint_dir, config_dir, size=512, old_version=False, preprocess='crop'):

    if old_version:
        #### load all the checkpoint of `pth`
        sadtalker_paths = {
                'wav2lip_checkpoint' : os.path.join(checkpoint_dir, 'wav2lip.pth'),
                'audio2pose_checkpoint' : os.path.join(checkpoint_dir, 'auido2pose_00140-model.pth'),
                'audio2exp_checkpoint' : os.path.join(checkpoint_dir, 'auido2exp_00300-model.pth'),
                'free_view_checkpoint' : os.path.join(checkpoint_dir, 'facevid2vid_00189-model.pth.tar'),
                'path_of_net_recon_model' : os.path.join(checkpoint_dir, 'epoch_20.pth')
        }

        use_safetensor = False
    elif len(glob.glob(os.path.join(checkpoint_dir, '*.safetensors'))):
        print('using safetensor as default')
        sadtalker_paths = {
            "checkpoint":os.path.join(checkpoint_dir, 'SadTalker_V0.0.2_'+str(size)+'.safetensors'),
            }
        use_safetensor = True
    else:
        print("WARNING: The new version of the model will be updated by safetensor, you may need to download it mannully. We run the old version of the checkpoint this time!")
        use_safetensor = False
        
        sadtalker_paths = {
                'wav2lip_checkpoint' : os.path.join(checkpoint_dir, 'wav2lip.pth'),
                'audio2pose_checkpoint' : os.path.join(checkpoint_dir, 'auido2pose_00140-model.pth'),
                'audio2exp_checkpoint' : os.path.join(checkpoint_dir, 'auido2exp_00300-model.pth'),
                'free_view_checkpoint' : os.path.join(checkpoint_dir, 'facevid2vid_00189-model.pth.tar'),
                'path_of_net_recon_model' : os.path.join(checkpoint_dir, 'epoch_20.pth')
        }

    sadtalker_paths['dir_of_BFM_fitting'] = os.path.join(config_dir) # , 'BFM_Fitting'
    sadtalker_paths['audio2pose_yaml_path'] = os.path.join(config_dir, 'auido2pose.yaml')
    sadtalker_paths['audio2exp_yaml_path'] = os.path.join(config_dir, 'auido2exp.yaml')
    sadtalker_paths['use_safetensor'] =  use_safetensor # os.path.join(config_dir, 'auido2exp.yaml')

    if 'full' in preprocess:
        sadtalker_paths['mappingnet_checkpoint'] = os.path.join(checkpoint_dir, 'mapping_00109-model.pth.tar')
        sadtalker_paths['facerender_yaml'] = os.path.join(config_dir, 'facerender_still.yaml')
    else:
        sadtalker_paths['mappingnet_checkpoint'] = os.path.join(checkpoint_dir, 'mapping_00229-model.pth.tar')
        sadtalker_paths['facerender_yaml'] = os.path.join(config_dir, 'facerender.yaml')

    return sadtalker_paths