import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from dependency_MMC import LABEL_COLUMN, MMC_LABEL_LIST


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
    try:
        return candidates.index(value)
    except ValueError as exc:
        raise ValueError(f'Unknown {LABEL_COLUMN}: {value}') from exc


def one_hot(index, num_classes):
    """Return a PyTorch one-hot vector without depending on TensorFlow/Keras."""
    return F.one_hot(torch.as_tensor(index, dtype=torch.long), num_classes=num_classes).float()


def encode_label(row):
    """Encode the MMC-AMD diagnosis label for one metadata row."""
    return _label_index(row[LABEL_COLUMN], MMC_LABEL_LIST)


def encode_meta_label(row):
    """Return metadata features for one row.

    The current MMC training loop does not consume tabular metadata, so this is
    intentionally an empty torch tensor. Keeping the function preserves the
    dataset return signature while avoiding the unused skin-lesion metadata
    encoders and the former TensorFlow/Keras one-hot dependency.
    """
    return torch.empty(0, dtype=torch.float32)
