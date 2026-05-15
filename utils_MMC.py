import os
import random

import numpy as np
import torch
from tensorflow.keras.utils import to_categorical

from dependency_MMC import *


def get_parameter_number(net):
    return {
        'Total': sum(p.numel() for p in net.parameters()),
        'Trainable': sum(p.numel() for p in net.parameters() if p.requires_grad),
    }


def create_cosine_learing_schdule(epochs, lr):
    return [float(lr / 2 * (np.cos(np.pi * epoch / epochs) + 1)) for epoch in range(epochs)]


class Logger:
    def open(self, name, mode):
        self.txt = open(name, mode=mode)

    def write(self, str_):
        self.txt.write(str_)
        self.txt.flush()
        print(str_, end='')

    def close(self):
        self.txt.close()


def set_seed(seed=15):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def CraateLogger(mode, model_name='resnet-50', round_=None, data_mode='Normal'):
    out_dir = './{}_{}_{}_weight_file/{}/'.format(mode, model_name, data_mode, round_)
    os.makedirs(out_dir + '/checkpoint/', exist_ok=True)
    os.makedirs(out_dir + '/train/', exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.single_modality_{}_skinlesion.txt'.format(mode), mode='w')
    return log, out_dir


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _label_index(value, candidates):
    for index, labels in enumerate(candidates):
        if value in labels:
            return index
    raise ValueError(f'Unknown label: {value}')


def _one_hot(value, candidates):
    return to_categorical(_label_index(value, candidates), len(candidates))


def encode_label(img_info, index_num):
    return np.array([
        _label_index(img_info['diagnosis'][index_num], label_list),
        _label_index(img_info['MMC_label'][index_num], MMC_label_list),
        _label_index(img_info['streaks'][index_num], streaks_label_list),
        _label_index(img_info['pigmentation'][index_num], pigmentation_label_list),
        _label_index(img_info['regression_structures'][index_num], regression_structures_label_list),
        _label_index(img_info['dots_and_globules'][index_num], dots_and_globules_label_list),
        _label_index(img_info['blue_whitish_veil'][index_num], blue_whitish_veil_label_list),
        _label_index(img_info['vascular_structures'][index_num], vascular_structures_label_list),
    ])


def encode_meta_label(img_info, index_num):
    return np.hstack([
        _one_hot(img_info['level_of_diagnostic_difficulty'][index_num], level_of_diagnostic_difficulty_label_list),
        _one_hot(img_info['elevation'][index_num], evaluation_list),
        _one_hot(img_info['location'][index_num], location_list),
        _one_hot(img_info['sex'][index_num], sex_list),
        _one_hot(img_info['management'][index_num], management_list),
    ])
