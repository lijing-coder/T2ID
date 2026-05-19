import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from dependency import (
    MMC_LABEL_TO_INDEX,
    img_info_path,
    source_dir,
    train_index_path,
    val_index_path,
)


try:
    from albumentations import (
        Compose,
        HorizontalFlip,
        RandomBrightnessContrast,
        RandomRotate90,
        ShiftScaleRotate,
        VerticalFlip,
    )

    train_aug = Compose(
        [
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.5,
                rotate_limit=45,
                p=0.5,
            ),
            RandomRotate90(p=0.5),
            RandomBrightnessContrast(p=0.5),
        ],
        p=0.5,
    )
except ImportError:
    train_aug = None


def load_image(path, shape):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.resize(image, shape)
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1))


class MMCDataset(Dataset):
    def __init__(self, image_dir, img_info, file_list, shape, is_train=True):
        self.image_dir = image_dir
        self.img_info = img_info
        self.file_list = list(file_list)
        self.shape = tuple(shape)
        self.is_train = is_train

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_id = int(self.file_list[index])
        row = self.img_info.loc[file_id]

        fundus_path = os.path.join(self.image_dir, row["CFP"])
        oct_path = os.path.join(self.image_dir, row["OCT"])

        fundus_img = cv2.imread(fundus_path)
        oct_img = cv2.imread(oct_path)
        if fundus_img is None:
            raise FileNotFoundError(f"Image not found: {fundus_path}")
        if oct_img is None:
            raise FileNotFoundError(f"Image not found: {oct_path}")

        fundus_img = cv2.resize(fundus_img, self.shape)
        oct_img = cv2.resize(oct_img, self.shape)

        if self.is_train and train_aug is not None:
            augmented = train_aug(image=fundus_img, mask=oct_img)
            fundus_img = augmented["image"]
            oct_img = augmented["mask"]

        fundus_img = torch.from_numpy(fundus_img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        oct_img = torch.from_numpy(oct_img.transpose(2, 0, 1).astype(np.float32) / 255.0)

        mmc_label = MMC_LABEL_TO_INDEX[row["MMC_label"]]
        meta_data = torch.empty(0)

        # Keep label[1] as the diagnosis label to remain compatible with the training script.
        label = [mmc_label, mmc_label]

        return fundus_img, oct_img, meta_data, label


def generate_dataloader(shape, batch_size, num_workers, data_mode="Normal"):
    img_info = pd.read_csv(img_info_path)
    train_indices = pd.read_csv(train_index_path)["indexes"].tolist()
    val_indices = pd.read_csv(val_index_path)["indexes"].tolist()

    if data_mode == "self_evaluated":
        train_indices = train_indices[:206]

    train_dataset = MMCDataset(
        image_dir=source_dir,
        img_info=img_info,
        file_list=train_indices,
        shape=shape,
        is_train=True,
    )
    val_dataset = MMCDataset(
        image_dir=source_dir,
        img_info=img_info,
        file_list=val_indices,
        shape=shape,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
