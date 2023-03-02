"""This script contains the training options for Deep3DFaceRecon_pytorch
"""

from .base_options import BaseOptions
from util import util

class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # dataset parameters
        # for train
        parser.add_argument('--data_root', type=str, default='./', help='dataset root')
        parser.add_argument('--flist', type=str, default='datalist/train/masks.txt', help='list of mask names of training set')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--dataset_mode', type=str, default='flist', help='chooses how datasets are loaded. [None | flist]')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='shift_scale_rot_flip', help='scaling and cropping of images at load time [shift_scale_rot_flip | shift_scale | shift | shift_rot_flip ]')
        parser.add_argument('--use_aug', type=util.str2bool, nargs='?', const=True, default=True, help='whether use data augmentation')

        # for val
        parser.add_argument('--flist_val', type=str, default='datalist/val/masks.txt', help='list of mask names of val set')
        parser.add_argument('--batch_size_val', type=int, default=32)


        # visualization parameters
        parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--evaluation_freq', type=int, default=5000, help='evaluation freq')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--pretrained_name', type=str, default=None, help='resume training from another checkpoint')

        # training parameters
        parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_epochs', type=int, default=10, help='multiply by a gamma every lr_decay_epochs epoches')

        self.isTrain = True
        return parser
