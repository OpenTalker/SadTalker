import sys
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm


class LayerNorm2d(nn.Module):
    def __init__(self, n_out, affine=True):
        super(LayerNorm2d, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
          self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
          self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
          return F.layer_norm(x, normalized_shape, \
              self.weight.expand(normalized_shape), 
              self.bias.expand(normalized_shape))
              
        else:
          return F.layer_norm(x, normalized_shape)  

class ADAINHourglass(nn.Module):
    def __init__(self, image_nc, pose_nc, ngf, img_f, encoder_layers, decoder_layers, nonlinearity, use_spect):
        super(ADAINHourglass, self).__init__()
        self.encoder = ADAINEncoder(image_nc, pose_nc, ngf, img_f, encoder_layers, nonlinearity, use_spect)
        self.decoder = ADAINDecoder(pose_nc, ngf, img_f, encoder_layers, decoder_layers, True, nonlinearity, use_spect)
        self.output_nc = self.decoder.output_nc

    def forward(self, x, z):
        return self.decoder(self.encoder(x, z), z)                 



class ADAINEncoder(nn.Module):
    def __init__(self, image_nc, pose_nc, ngf, img_f, layers, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINEncoder, self).__init__()
        self.layers = layers
        self.input_layer = nn.Conv2d(image_nc, ngf, kernel_size=7, stride=1, padding=3)
        for i in range(layers):
            in_channels = min(ngf * (2**i), img_f)
            out_channels = min(ngf *(2**(i+1)), img_f)
            model = ADAINEncoderBlock(in_channels, out_channels, pose_nc, nonlinearity, use_spect)
            setattr(self, 'encoder' + str(i), model)
        self.output_nc = out_channels
        
    def forward(self, x, z):
        out = self.input_layer(x)
        out_list = [out]
        for i in range(self.layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out, z)
            out_list.append(out)
        return out_list
        
class ADAINDecoder(nn.Module):
    """docstring for ADAINDecoder"""
    def __init__(self, pose_nc, ngf, img_f, encoder_layers, decoder_layers, skip_connect=True, 
                 nonlinearity=nn.LeakyReLU(), use_spect=False):

        super(ADAINDecoder, self).__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.skip_connect = skip_connect
        use_transpose = True

        for i in range(encoder_layers-decoder_layers, encoder_layers)[::-1]:
            in_channels = min(ngf * (2**(i+1)), img_f)
            in_channels = in_channels*2 if i != (encoder_layers-1) and self.skip_connect else in_channels
            out_channels = min(ngf * (2**i), img_f)
            model = ADAINDecoderBlock(in_channels, out_channels, out_channels, pose_nc, use_transpose, nonlinearity, use_spect)
            setattr(self, 'decoder' + str(i), model)

        self.output_nc = out_channels*2 if self.skip_connect else out_channels

    def forward(self, x, z):
        out = x.pop() if self.skip_connect else x
        for i in range(self.encoder_layers-self.decoder_layers, self.encoder_layers)[::-1]:
            model = getattr(self, 'decoder' + str(i))
            out = model(out, z)
            out = torch.cat([out, x.pop()], 1) if self.skip_connect else out
        return out

class ADAINEncoderBlock(nn.Module):       
    def __init__(self, input_nc, output_nc, feature_nc, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINEncoderBlock, self).__init__()
        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        self.conv_0 = spectral_norm(nn.Conv2d(input_nc,  output_nc, **kwargs_down), use_spect)
        self.conv_1 = spectral_norm(nn.Conv2d(output_nc, output_nc, **kwargs_fine), use_spect)


        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(output_nc, feature_nc)
        self.actvn = nonlinearity

    def forward(self, x, z):
        x = self.conv_0(self.actvn(self.norm_0(x, z)))
        x = self.conv_1(self.actvn(self.norm_1(x, z)))
        return x

class ADAINDecoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc, feature_nc, use_transpose=True, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINDecoderBlock, self).__init__()        
        # Attributes
        self.actvn = nonlinearity
        hidden_nc = min(input_nc, output_nc) if hidden_nc is None else hidden_nc

        kwargs_fine = {'kernel_size':3, 'stride':1, 'padding':1}
        if use_transpose:
            kwargs_up = {'kernel_size':3, 'stride':2, 'padding':1, 'output_padding':1}
        else:
            kwargs_up = {'kernel_size':3, 'stride':1, 'padding':1}

        # create conv layers
        self.conv_0 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, **kwargs_fine), use_spect)
        if use_transpose:
            self.conv_1 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, **kwargs_up), use_spect)
            self.conv_s = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, **kwargs_up), use_spect)
        else:
            self.conv_1 = nn.Sequential(spectral_norm(nn.Conv2d(hidden_nc, output_nc, **kwargs_up), use_spect),
                                        nn.Upsample(scale_factor=2))
            self.conv_s = nn.Sequential(spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs_up), use_spect),
                                        nn.Upsample(scale_factor=2))
        # define normalization layers
        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(hidden_nc, feature_nc)
        self.norm_s = ADAIN(input_nc, feature_nc)
        
    def forward(self, x, z):
        x_s = self.shortcut(x, z)
        dx = self.conv_0(self.actvn(self.norm_0(x, z)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, z)))
        out = x_s + dx
        return out

    def shortcut(self, x, z):
        x_s = self.conv_s(self.actvn(self.norm_s(x, z)))
        return x_s              


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module


class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta
        return out


class FineEncoder(nn.Module):
    """docstring for Encoder"""
    def __init__(self, image_nc, ngf, img_f, layers, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineEncoder, self).__init__()
        self.layers = layers
        self.first = FirstBlock2d(image_nc, ngf, norm_layer, nonlinearity, use_spect)
        for i in range(layers):
            in_channels = min(ngf*(2**i), img_f)
            out_channels = min(ngf*(2**(i+1)), img_f)
            model = DownBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            setattr(self, 'down' + str(i), model)
        self.output_nc = out_channels

    def forward(self, x):
        x = self.first(x)
        out=[x]
        for i in range(self.layers):
            model = getattr(self, 'down'+str(i))
            x = model(x)
            out.append(x)
        return out

class FineDecoder(nn.Module):
    """docstring for FineDecoder"""
    def __init__(self, image_nc, feature_nc, ngf, img_f, layers, num_block, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineDecoder, self).__init__()
        self.layers = layers
        for i in range(layers)[::-1]:
            in_channels = min(ngf*(2**(i+1)), img_f)
            out_channels = min(ngf*(2**i), img_f)
            up = UpBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            res = FineADAINResBlocks(num_block, in_channels, feature_nc, norm_layer, nonlinearity, use_spect)
            jump = Jump(out_channels, norm_layer, nonlinearity, use_spect)

            setattr(self, 'up' + str(i), up)
            setattr(self, 'res' + str(i), res)            
            setattr(self, 'jump' + str(i), jump)

        self.final = FinalBlock2d(out_channels, image_nc, use_spect, 'tanh')

        self.output_nc = out_channels

    def forward(self, x, z):
        out = x.pop()
        for i in range(self.layers)[::-1]:
            res_model = getattr(self, 'res' + str(i))
            up_model = getattr(self, 'up' + str(i))
            jump_model = getattr(self, 'jump' + str(i))
            out = res_model(out, z)
            out = up_model(out)
            out = jump_model(x.pop()) + out
        out_image = self.final(out)
        return out_image

class FirstBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FirstBlock2d, self).__init__()
        kwargs = {'kernel_size': 7, 'stride': 1, 'padding': 3}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity)


    def forward(self, x):
        out = self.model(x)
        return out  

class DownBlock2d(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(DownBlock2d, self).__init__()


        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)
        pool = nn.AvgPool2d(kernel_size=(2, 2))

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity, pool)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity, pool)

    def forward(self, x):
        out = self.model(x)
        return out 

class UpBlock2d(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(UpBlock2d, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)
        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity)

    def forward(self, x):
        out = self.model(F.interpolate(x, scale_factor=2))
        return out

class FineADAINResBlocks(nn.Module):
    def __init__(self, num_block, input_nc, feature_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineADAINResBlocks, self).__init__()                                
        self.num_block = num_block
        for i in range(num_block):
            model = FineADAINResBlock2d(input_nc, feature_nc, norm_layer, nonlinearity, use_spect)
            setattr(self, 'res'+str(i), model)

    def forward(self, x, z):
        for i in range(self.num_block):
            model = getattr(self, 'res'+str(i))
            x = model(x, z)
        return x     

class Jump(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(Jump, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(input_nc), nonlinearity)

    def forward(self, x):
        out = self.model(x)
        return out          

class FineADAINResBlock2d(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, feature_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineADAINResBlock2d, self).__init__()

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        self.conv1 = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)
        self.conv2 = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)
        self.norm1 = ADAIN(input_nc, feature_nc)
        self.norm2 = ADAIN(input_nc, feature_nc)

        self.actvn = nonlinearity


    def forward(self, x, z):
        dx = self.actvn(self.norm1(self.conv1(x), z))
        dx = self.norm2(self.conv2(x), z)
        out = dx + x
        return out        

class FinalBlock2d(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, use_spect=False, tanh_or_sigmoid='tanh'):
        super(FinalBlock2d, self).__init__()

        kwargs = {'kernel_size': 7, 'stride': 1, 'padding':3}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

        if tanh_or_sigmoid == 'sigmoid':
            out_nonlinearity = nn.Sigmoid()
        else:
            out_nonlinearity = nn.Tanh()            

        self.model = nn.Sequential(conv, out_nonlinearity)
    def forward(self, x):
        out = self.model(x)
        return out          