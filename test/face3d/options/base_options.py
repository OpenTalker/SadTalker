"""This script contains base options for Deep3DFaceRecon_pytorch
"""

import argparse
import os
from util import util
import numpy as np
import torch
import face3d.models as models
import face3d.data as data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, cmd_line=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='face_recon', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--vis_batch_nums', type=float, default=1, help='batch nums of images for visulization')
        parser.add_argument('--eval_batch_nums', type=float, default=float('inf'), help='batch nums of images for evaluation')
        parser.add_argument('--use_ddp', type=util.str2bool, nargs='?', const=True, default=True, help='whether use distributed data parallel')
        parser.add_argument('--ddp_port', type=str, default='12355', help='ddp port')
        parser.add_argument('--display_per_batch', type=util.str2bool, nargs='?', const=True, default=True, help='whether use batch to show losses')
        parser.add_argument('--add_image', type=util.str2bool, nargs='?', const=True, default=True, help='whether add image to tensorboard')
        parser.add_argument('--world_size', type=int, default=1, help='batch nums of images for evaluation')

        # model parameters
        parser.add_argument('--model', type=str, default='facerecon', help='chooses which model to use.')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # set cuda visible devices
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()  # parse again with new defaults
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)  # parse again with new defaults

        # modify dataset-related parser options
        if opt.dataset_mode:
            dataset_name = opt.dataset_mode
            dataset_option_setter = data.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        try:
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        except PermissionError as error:
            print("permission error {}".format(error))
            pass

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix


        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        opt.world_size = len(gpu_ids)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(gpu_ids[0])
        if opt.world_size == 1:
            opt.use_ddp = False

        if opt.phase != 'test':
            # set continue_train automatically
            if opt.pretrained_name is None:
                model_dir = os.path.join(opt.checkpoints_dir, opt.name)
            else:
                model_dir = os.path.join(opt.checkpoints_dir, opt.pretrained_name)
            if os.path.isdir(model_dir):
                model_pths = [i for i in os.listdir(model_dir) if i.endswith('pth')]
                if os.path.isdir(model_dir) and len(model_pths) != 0:
                    opt.continue_train= True
        
            # update the latest epoch count
            if opt.continue_train:
                if opt.epoch == 'latest':
                    epoch_counts = [int(i.split('.')[0].split('_')[-1]) for i in model_pths if 'latest' not in i]
                    if len(epoch_counts) != 0:
                        opt.epoch_count = max(epoch_counts) + 1
                else:
                    opt.epoch_count = int(opt.epoch) + 1
                    

        self.print_options(opt)
        self.opt = opt
        return self.opt
