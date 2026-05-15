import os

import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from dependency_MMC import (
    FUNDUS_IMAGE_COLUMN,
    INDEX_COLUMN,
    OCT_IMAGE_COLUMN,
    img_info_path,
    source_dir,
    train_index_path,
    val_index_path,
)
from utils_MMC import encode_label, encode_meta_label


def _resolve_image_path(image_dir, image_name):
    image_name = str(image_name)
    if os.path.isabs(image_name):
        return image_name
    return os.path.join(image_dir, image_name)


def load_image(path, shape):
    """Read and resize an image as a float32 torch tensor in CHW/RGB format."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Image file does not exist: {path}')

    image = read_image(path, mode=ImageReadMode.RGB).float().div(255.0)
    height, width = shape[1], shape[0]
    return TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR, antialias=True)


def augment_pair(fundus_img, oct_img):
    """Apply paired torch augmentations to keep the two modalities aligned."""
    if torch.rand(()) >= 0.5:
        return fundus_img, oct_img

    if torch.rand(()) < 0.5:
        fundus_img = TF.vflip(fundus_img)
        oct_img = TF.vflip(oct_img)

    if torch.rand(()) < 0.5:
        fundus_img = TF.hflip(fundus_img)
        oct_img = TF.hflip(oct_img)

    if torch.rand(()) < 0.5:
        angle = float(torch.empty(1).uniform_(-45, 45).item())
        max_dx = 0.0625 * fundus_img.shape[2]
        max_dy = 0.0625 * fundus_img.shape[1]
        translate = [
            int(torch.empty(1).uniform_(-max_dx, max_dx).item()),
            int(torch.empty(1).uniform_(-max_dy, max_dy).item()),
        ]
        scale = float(torch.empty(1).uniform_(0.5, 1.5).item())
        fundus_img = TF.affine(
            fundus_img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        oct_img = TF.affine(
            oct_img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

    if torch.rand(()) < 0.5:
        k = int(torch.randint(1, 4, (1,)).item())
        fundus_img = torch.rot90(fundus_img, k, dims=(1, 2))
        oct_img = torch.rot90(oct_img, k, dims=(1, 2))

    if torch.rand(()) < 0.5:
        brightness = float(torch.empty(1).uniform_(0.8, 1.2).item())
        contrast = float(torch.empty(1).uniform_(0.8, 1.2).item())
        fundus_img = TF.adjust_contrast(TF.adjust_brightness(fundus_img, brightness), contrast).clamp(0, 1)
        oct_img = TF.adjust_contrast(TF.adjust_brightness(oct_img, brightness), contrast).clamp(0, 1)

    return fundus_img, oct_img


class SkinDataset(data.Dataset):
    def __init__(self, image_dir, img_info, file_list, shape, is_test=False):
        self.is_test = is_test
        self.image_dir = image_dir
        self.img_info = img_info.reset_index(drop=True)
        self.file_list = list(file_list)
        self.shape = shape

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(len(self.file_list)):
            return self.__getitem__(torch.randint(len(self), (1,)).item())

        file_id = int(self.file_list[index])
        row = self.img_info.iloc[file_id]

        fundus_img = load_image(_resolve_image_path(self.image_dir, row[FUNDUS_IMAGE_COLUMN]), self.shape)
        oct_img = load_image(_resolve_image_path(self.image_dir, row[OCT_IMAGE_COLUMN]), self.shape)

        if not self.is_test:
            fundus_img, oct_img = augment_pair(fundus_img, oct_img)

        meta_data = encode_meta_label(row)
        diagnosis_label = torch.tensor(encode_label(row), dtype=torch.long)

        return fundus_img, oct_img, meta_data, diagnosis_label


def _build_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
    )


def generate_dataloader(shape, batch_size, num_workers, data_mode):
    train_index_list = list(pd.read_csv(train_index_path)[INDEX_COLUMN])
    val_index_list = list(pd.read_csv(val_index_path)[INDEX_COLUMN])
    img_info = pd.read_csv(img_info_path)

    if data_mode == 'self_evaluated':
        train_index_list = train_index_list[:206]

    train_dataset = SkinDataset(source_dir, img_info, train_index_list, shape, is_test=False)
    val_dataset = SkinDataset(source_dir, img_info, val_index_list, shape, is_test=True)

    train_dataloader = _build_loader(train_dataset, batch_size, num_workers, shuffle=True)
    val_dataloader = _build_loader(val_dataset, batch_size, num_workers, shuffle=True)
    return train_dataloader, val_dataloader
