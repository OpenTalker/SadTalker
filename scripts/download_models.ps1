# Create directory
mkdir ".\checkpoints"

# Legacy download links
# Invoke-WebRequest -Uri "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/auido2exp_00300-model.pth" -OutFile ".\checkpoints\auido2exp_00300-model.pth"
# Invoke-WebRequest -Uri "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/auido2pose_00140-model.pth" -OutFile ".\checkpoints\auido2pose_00140-model.pth"
# Invoke-WebRequest -Uri "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/epoch_20.pth" -OutFile ".\checkpoints\epoch_20.pth"
# Invoke-WebRequest -Uri "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/facevid2vid_00189-model.pth.tar" -OutFile ".\checkpoints\facevid2vid_00189-model.pth.tar"
# Invoke-WebRequest -Uri "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/shape_predictor_68_face_landmarks.dat" -OutFile ".\checkpoints\shape_predictor_68_face_landmarks.dat"
# Invoke-WebRequest -Uri "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/wav2lip.pth" -OutFile ".\checkpoints\wav2lip.pth"
# Invoke-WebRequest -Uri "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/mapping_00229-model.pth.tar" -OutFile ".\checkpoints\mapping_00229-model.pth.tar"
# Invoke-WebRequest -Uri "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/mapping_00109-model.pth.tar" -OutFile ".\checkpoints\mapping_00109-model.pth.tar"
# Invoke-WebRequest -Uri "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/hub.zip" -OutFile ".\checkpoints\hub.zip"
# Expand-Archive -Path ".\checkpoints\hub.zip" -DestinationPath ".\checkpoints\" -Force

# Download the new links
Invoke-WebRequest -Uri "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar" -OutFile ".\checkpoints\mapping_00109-model.pth.tar"
Invoke-WebRequest -Uri "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar" -OutFile ".\checkpoints\mapping_00229-model.pth.tar"
Invoke-WebRequest -Uri "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors" -OutFile ".\checkpoints\SadTalker_V0.0.2_256.safetensors"
Invoke-WebRequest -Uri "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors" -OutFile ".\checkpoints\SadTalker_V0.0.2_512.safetensors"

# Invoke-WebRequest -Uri "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip" -OutFile ".\checkpoints\BFM_Fitting.zip"
# Expand-Archive -Path ".\checkpoints\BFM_Fitting.zip" -DestinationPath ".\checkpoints\" -Force

# Enhancer
mkdir ".\gfpgan\weights"
Invoke-WebRequest -Uri "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth" -OutFile ".\gfpgan\weights\alignment_WFLW_4HG.pth"
Invoke-WebRequest -Uri "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" -OutFile ".\gfpgan\weights\detection_Resnet50_Final.pth"
Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" -OutFile ".\gfpgan\weights\GFPGANv1.4.pth"
Invoke-WebRequest -Uri "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" -OutFile ".\gfpgan\weights\parsing_parsenet.pth"
