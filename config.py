from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.CONFIG_NAME = ''
__C.GPU_ID = '0'
__C.CUDA = True
__C.WORKERS = 4


# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 10
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.FLAG = True
__C.TRAIN.NET_G = ''
__C.TRAIN.NET_D = ''
__C.TRAIN.FINE_TUNE = False
__C.TRAIN.AVERAGE = False

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.ADV_LOSS = 1.0
__C.TRAIN.COEFF.CONT_LOSS = 1.0
__C.TRAIN.COEFF.CONT_TEMP = 1.0


# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 64

# fMRI options
__C.BRAIN = edict()
__C.BRAIN.TRIAL_NUM = 5
__C.BRAIN.DIMENSION = 7115


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)
