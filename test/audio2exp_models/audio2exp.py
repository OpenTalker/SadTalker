import torch
from torch import nn


class Audio2Exp(nn.Module):
    def __init__(self, netG, cfg, device, prepare_training_loss=False):
        super(Audio2Exp, self).__init__()
        self.cfg = cfg
        self.device = device
        self.netG = netG.to(device)

    def test(self, batch):

        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]

        ref = batch['ref'][:, :, :64].repeat((1,T,1))           #bs T 64
        ratio = batch['ratio_gt']                               #bs T

        audiox = mel_input.view(-1, 1, 80, 16)                  # bs*T 1 80 16
        exp_coeff_pred  = self.netG(audiox, ref, ratio)         # bs T 64 

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': exp_coeff_pred
            }
        return results_dict


