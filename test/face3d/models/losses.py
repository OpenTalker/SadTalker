import numpy as np
import torch
import torch.nn as nn
from kornia.geometry import warp_affine
import torch.nn.functional as F

def resize_n_crop(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return warp_affine(image, M, dsize=(dsize, dsize), align_corners=True)

### perceptual level loss
class PerceptualLoss(nn.Module):
    def __init__(self, recog_net, input_size=112):
        super(PerceptualLoss, self).__init__()
        self.recog_net = recog_net
        self.preprocess = lambda x: 2 * x - 1
        self.input_size=input_size
    def forward(imageA, imageB, M):
        """
        1 - cosine distance
        Parameters:
            imageA       --torch.tensor (B, 3, H, W), range (0, 1) , RGB order
            imageB       --same as imageA
        """

        imageA = self.preprocess(resize_n_crop(imageA, M, self.input_size))
        imageB = self.preprocess(resize_n_crop(imageB, M, self.input_size))

        # freeze bn
        self.recog_net.eval()
        
        id_featureA = F.normalize(self.recog_net(imageA), dim=-1, p=2)
        id_featureB = F.normalize(self.recog_net(imageB), dim=-1, p=2)  
        cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
        # assert torch.sum((cosine_d > 1).float()) == 0
        return torch.sum(1 - cosine_d) / cosine_d.shape[0]        

def perceptual_loss(id_featureA, id_featureB):
    cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
        # assert torch.sum((cosine_d > 1).float()) == 0
    return torch.sum(1 - cosine_d) / cosine_d.shape[0]  

### image level loss
def photo_loss(imageA, imageB, mask, eps=1e-6):
    """
    l2 norm (with sqrt, to ensure backward stabililty, use eps, otherwise Nan may occur)
    Parameters:
        imageA       --torch.tensor (B, 3, H, W), range (0, 1), RGB order 
        imageB       --same as imageA
    """
    loss = torch.sqrt(eps + torch.sum((imageA - imageB) ** 2, dim=1, keepdims=True)) * mask
    loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    return loss

def landmark_loss(predict_lm, gt_lm, weight=None):
    """
    weighted mse loss
    Parameters:
        predict_lm    --torch.tensor (B, 68, 2)
        gt_lm         --torch.tensor (B, 68, 2)
        weight        --numpy.array (1, 68)
    """
    if not weight:
        weight = np.ones([68])
        weight[28:31] = 20
        weight[-8:] = 20
        weight = np.expand_dims(weight, 0)
        weight = torch.tensor(weight).to(predict_lm.device)
    loss = torch.sum((predict_lm - gt_lm)**2, dim=-1) * weight
    loss = torch.sum(loss) / (predict_lm.shape[0] * predict_lm.shape[1])
    return loss


### regulization
def reg_loss(coeffs_dict, opt=None):
    """
    l2 norm without the sqrt, from yu's implementation (mse)
    tf.nn.l2_loss https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    Parameters:
        coeffs_dict     -- a  dict of torch.tensors , keys: id, exp, tex, angle, gamma, trans

    """
    # coefficient regularization to ensure plausible 3d faces
    if opt:
        w_id, w_exp, w_tex = opt.w_id, opt.w_exp, opt.w_tex
    else:
        w_id, w_exp, w_tex = 1, 1, 1, 1
    creg_loss = w_id * torch.sum(coeffs_dict['id'] ** 2) +  \
           w_exp * torch.sum(coeffs_dict['exp'] ** 2) + \
           w_tex * torch.sum(coeffs_dict['tex'] ** 2)
    creg_loss = creg_loss / coeffs_dict['id'].shape[0]

    # gamma regularization to ensure a nearly-monochromatic light
    gamma = coeffs_dict['gamma'].reshape([-1, 3, 9])
    gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
    gamma_loss = torch.mean((gamma - gamma_mean) ** 2)

    return creg_loss, gamma_loss

def reflectance_loss(texture, mask):
    """
    minimize texture variance (mse), albedo regularization to ensure an uniform skin albedo
    Parameters:
        texture       --torch.tensor, (B, N, 3)
        mask          --torch.tensor, (N), 1 or 0

    """
    mask = mask.reshape([1, mask.shape[0], 1])
    texture_mean = torch.sum(mask * texture, dim=1, keepdims=True) / torch.sum(mask)
    loss = torch.sum(((texture - texture_mean) * mask)**2) / (texture.shape[0] * torch.sum(mask))
    return loss

