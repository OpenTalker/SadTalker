mkdir ./checkpoints  
wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.1/auido2exp_00300-model.pth -O ./checkpoints/auido2exp_00300-model.pth
wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.1/auido2pose_00140-model.pth -O ./checkpoints/auido2pose_00140-model.pth
wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.1/epoch_20.pth -O ./checkpoints/epoch_20.pth
wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.1/facevid2vid_00189-model.pth.tar -O ./checkpoints/facevid2vid_00189-model.pth.tar
wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat -O ./checkpoints/shape_predictor_68_face_landmarks.dat
wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.1/wav2lip.pth -O ./checkpoints/wav2lip.pth
wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.1/mapping_00229-model.pth.tar -O ./checkpoints/mapping_00229-model.pth.tar
wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.1/BFM_Fitting.zip -O ./checkpoints/BFM_Fitting.zip
wget -nc https://github.com/Winfredy/SadTalker/releases/download/v0.0.1/hub.zip -O ./checkpoints/hub.zip

unzip -n ./checkpoints/hub.zip -d ./checkpoints/
unzip -n ./checkpoints/BFM_Fitting.zip -d ./checkpoints/