import torch
from torch import nn
from torch.nn import functional as F

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class AudioEncoder(nn.Module):
    def __init__(self, wav2lip_checkpoint, device):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        #### load the pre-trained audio_encoder, we do not need to load wav2lip model here.
        # wav2lip_state_dict = torch.load(wav2lip_checkpoint, map_location=torch.device(device))['state_dict']
        # state_dict = self.audio_encoder.state_dict()

        # for k,v in wav2lip_state_dict.items():
        #     if 'audio_encoder' in k:
        #         state_dict[k.replace('module.audio_encoder.', '')] = v
        # self.audio_encoder.load_state_dict(state_dict)


    def forward(self, audio_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1
        dim = audio_embedding.shape[1]
        audio_embedding = audio_embedding.reshape((B, -1, dim, 1, 1))

        return audio_embedding.squeeze(-1).squeeze(-1) #B seq_len+1 512 
