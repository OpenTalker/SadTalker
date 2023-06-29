import collections
import functools
import os
import re

import yaml

class AttrDict(dict):
    """Dict as attribute trick."""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, (list, tuple)):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """Convert object to yaml dict and return."""
        yaml_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables."""
        ret_str = []
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in value:
                        # Treat as AttrDict above.
                        child_ret_str = item.__repr__().split('\n')
                        for item in child_ret_str:
                            ret_str.append('    ' + item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)


class Config(AttrDict):
    r"""Configuration class. This should include every human specifiable
    hyperparameter values for your training."""

    def __init__(self, filename=None, args=None, verbose=False, is_train=True):
        super(Config, self).__init__()
        # Set default parameters.
        # Logging.

        large_number = 1000000000
        self.snapshot_save_iter = large_number
        self.snapshot_save_epoch = large_number
        self.snapshot_save_start_iter = 0
        self.snapshot_save_start_epoch = 0
        self.image_save_iter = large_number
        self.eval_epoch = large_number
        self.start_eval_epoch = large_number
        self.eval_epoch = large_number
        self.max_epoch = large_number
        self.max_iter = large_number
        self.logging_iter = 100
        self.image_to_tensorboard=False
        self.which_iter = 0 # args.which_iter
        self.resume = False

        self.checkpoints_dir = '/Users/shadowcun/Downloads/'
        self.name = 'face'
        self.phase = 'train' if is_train else 'test'

        # Networks.
        self.gen = AttrDict(type='generators.dummy')
        self.dis = AttrDict(type='discriminators.dummy')

        # Optimizers.
        self.gen_optimizer = AttrDict(type='adam',
                                    lr=0.0001,
                                    adam_beta1=0.0,
                                    adam_beta2=0.999,
                                    eps=1e-8,
                                    lr_policy=AttrDict(iteration_mode=False,
                                                    type='step',
                                                    step_size=large_number,
                                                    gamma=1))
        self.dis_optimizer = AttrDict(type='adam',
                                lr=0.0001,
                                adam_beta1=0.0,
                                adam_beta2=0.999,
                                eps=1e-8,
                                lr_policy=AttrDict(iteration_mode=False,
                                                   type='step',
                                                   step_size=large_number,
                                                   gamma=1))
        # Data.
        self.data = AttrDict(name='dummy',
                             type='datasets.images',
                             num_workers=0)
        self.test_data = AttrDict(name='dummy',
                                  type='datasets.images',
                                  num_workers=0,
                                  test=AttrDict(is_lmdb=False,
                                                roots='',
                                                batch_size=1))
        self.trainer = AttrDict(
            model_average=False,
            model_average_beta=0.9999,
            model_average_start_iteration=1000,
            model_average_batch_norm_estimation_iteration=30,
            model_average_remove_sn=True,
            image_to_tensorboard=False,
            hparam_to_tensorboard=False,
            distributed_data_parallel='pytorch',
            delay_allreduce=True,
            gan_relativistic=False,
            gen_step=1,
            dis_step=1)

        # # Cudnn.
        self.cudnn = AttrDict(deterministic=False,
                              benchmark=True)

        # Others.
        self.pretrained_weight = ''
        self.inference_args = AttrDict()


        # Update with given configurations.
        assert os.path.exists(filename), 'File {} not exist.'.format(filename)
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        try:
            with open(filename, 'r') as f:
                cfg_dict = yaml.load(f, Loader=loader)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        recursive_update(self, cfg_dict)

        # Put common opts in both gen and dis.
        if 'common' in cfg_dict:
            self.common = AttrDict(**cfg_dict['common'])
            self.gen.common = self.common
            self.dis.common = self.common


        if verbose:
            print(' config '.center(80, '-'))
            print(self.__repr__())
            print(''.center(80, '-'))


def rsetattr(obj, attr, val):
    """Recursively find object and set value"""
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """Recursively find object and return value"""

    def _getattr(obj, attr):
        r"""Get attribute."""
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def recursive_update(d, u):
    """Recursively update AttrDict d with AttrDict u"""
    for key, value in u.items():
        if isinstance(value, collections.abc.Mapping):
            d.__dict__[key] = recursive_update(d.get(key, AttrDict({})), value)
        elif isinstance(value, (list, tuple)):
            if isinstance(value[0], dict):
                d.__dict__[key] = [AttrDict(item) for item in value]
            else:
                d.__dict__[key] = value
        else:
            d.__dict__[key] = value
    return d
