from torch import nn
import torch
import torch.nn.functional as F

from src.facerender.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from src.facerender.modules.util import KPHourglass, make_coordinate_grid, AntiAliasInterpolation2d, ResBottleneck


class KPDetector(nn.Module):
    """
    Detecting canonical keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, reshape_channel, reshape_depth,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1, single_jacobian_map=False):
        super(KPDetector, self).__init__()

        self.predictor = KPHourglass(block_expansion, in_features=image_channel,
                                     max_features=max_features,  reshape_features=reshape_channel, reshape_depth=reshape_depth, num_blocks=num_blocks)

        # self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=7, padding=3)
        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=3, padding=1)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            # self.jacobian = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=9 * self.num_jacobian_maps, kernel_size=7, padding=3)
            self.jacobian = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=9 * self.num_jacobian_maps, kernel_size=3, padding=1)
            '''
            initial as:
            [[1 0 0]
             [0 1 0]
             [0 0 1]]
            '''
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(image_channel, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3, 4))
        kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 9, final_shape[2],
                                                final_shape[3], final_shape[4])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 9, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 3, 3)
            out['jacobian'] = jacobian

        return out


class HEEstimator(nn.Module):
    """
    Estimating head pose and expression.
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, num_bins=66, estimate_jacobian=True):
        super(HEEstimator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=image_channel, out_channels=block_expansion, kernel_size=7, padding=3, stride=2)
        self.norm1 = BatchNorm2d(block_expansion, affine=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=block_expansion, out_channels=256, kernel_size=1)
        self.norm2 = BatchNorm2d(256, affine=True)

        self.block1 = nn.Sequential()
        for i in range(3):
            self.block1.add_module('b1_'+ str(i), ResBottleneck(in_features=256, stride=1))

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.norm3 = BatchNorm2d(512, affine=True)
        self.block2 = ResBottleneck(in_features=512, stride=2)

        self.block3 = nn.Sequential()
        for i in range(3):
            self.block3.add_module('b3_'+ str(i), ResBottleneck(in_features=512, stride=1))

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.norm4 = BatchNorm2d(1024, affine=True)
        self.block4 = ResBottleneck(in_features=1024, stride=2)

        self.block5 = nn.Sequential()
        for i in range(5):
            self.block5.add_module('b5_'+ str(i), ResBottleneck(in_features=1024, stride=1))

        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        self.norm5 = BatchNorm2d(2048, affine=True)
        self.block6 = ResBottleneck(in_features=2048, stride=2)

        self.block7 = nn.Sequential()
        for i in range(2):
            self.block7.add_module('b7_'+ str(i), ResBottleneck(in_features=2048, stride=1))

        self.fc_roll = nn.Linear(2048, num_bins)
        self.fc_pitch = nn.Linear(2048, num_bins)
        self.fc_yaw = nn.Linear(2048, num_bins)

        self.fc_t = nn.Linear(2048, 3)

        self.fc_exp = nn.Linear(2048, 3*num_kp)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)

        out = self.block1(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.block2(out)

        out = self.block3(out)

        out = self.conv4(out)
        out = self.norm4(out)
        out = F.relu(out)
        out = self.block4(out)

        out = self.block5(out)

        out = self.conv5(out)
        out = self.norm5(out)
        out = F.relu(out)
        out = self.block6(out)

        out = self.block7(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.shape[0], -1)

        yaw = self.fc_roll(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_yaw(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}

