import os
import random

import numpy as np
import torch

from dependency_MMC import MMC_LABEL_TO_INDEX


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_cosine_learing_schdule(epochs, lr):
    return [
        float(lr * 0.5 * (1.0 + np.cos(np.pi * epoch / epochs)))
        for epoch in range(epochs)
    ]


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class Logger:
    def open(self, path, mode="w"):
        self.txt = open(path, mode)

    def write(self, message):
        self.txt.write(message)
        self.txt.flush()
        print(message, end="" if message.endswith("\n") else "\n")

    def close(self):
        self.txt.close()


def CraateLogger(mode, model_name="model", round_=0, data_mode="Normal"):
    out_dir = f"./{mode}_{model_name}_{data_mode}_weight_file/{round_}"
    checkpoint_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log = Logger()
    log.open(os.path.join(out_dir, f"log_{mode}.txt"), mode="w")
    log.write(f"--- Start training ---\n")
    log.write(f"out_dir: {out_dir}\n")
    log.write(f"mode: {mode}\n")
    log.write(f"model_name: {model_name}\n")
    log.write(f"data_mode: {data_mode}\n\n")

    return log, out_dir


def encode_label(img_info, index_num):
    row = img_info.loc[index_num]
    mmc_label = MMC_LABEL_TO_INDEX[row["MMC_label"]]
    return np.array([mmc_label, mmc_label])


def encode_test_label(img_info, index_num):
    label = encode_label(img_info, index_num)
    return label


def encode_meta_label(img_info, index_num):
    return np.empty(0, dtype=np.float32)
