import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, RandomRotate90, ShiftScaleRotate, VerticalFlip
from torch.utils import data
from torch.utils.data import DataLoader

from dependency_MMC import img_info_path, source_dir, train_index_path, val_index_path
from utils_MMC import encode_label, encode_meta_label


aug = Compose([
    VerticalFlip(p=0.5),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, p=0.5),
    RandomRotate90(p=0.5),
    RandomBrightnessContrast(p=0.5),
], p=0.5)


def load_image(path, shape):
    img = cv2.imread(path)
    return cv2.resize(img, (shape[0], shape[1]))


class SkinDataset(data.Dataset):
    def __init__(self, image_dir, img_info, file_list, shape, is_test=False, num_class=1):
        self.is_test = is_test
        self.image_dir = image_dir
        self.img_info = img_info
        self.file_list = file_list
        self.shape = shape
        self.num_class = num_class
        self.total_img_info = img_info

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(len(self.file_list)):
            return self.__getitem__(np.random.randint(0, len(self)))

        file_id = self.file_list[index]
        sub_img_info = self.total_img_info[file_id:file_id + 1]

        clinic_img = load_image(self.image_dir + sub_img_info['CFP'][file_id], self.shape)
        dermoscopy_img = load_image(self.image_dir + sub_img_info['OCT'][file_id], self.shape)

        if not self.is_test:
            augmented = aug(image=clinic_img, mask=dermoscopy_img)
            clinic_img = augmented['image']
            dermoscopy_img = augmented['mask']

        total_label = encode_label(sub_img_info, file_id)
        clinic_img = torch.from_numpy(np.transpose(clinic_img, (2, 0, 1)).astype('float32') / 255)
        dermoscopy_img = torch.from_numpy(np.transpose(dermoscopy_img, (2, 0, 1)).astype('float32') / 255)
        meta_data = encode_meta_label(sub_img_info, file_id)

        return clinic_img, dermoscopy_img, meta_data, [
            total_label[0], total_label[1], total_label[2], total_label[3],
            total_label[4], total_label[5], total_label[6], total_label[7]
        ]


def _build_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
    )


def generate_dataloader(shape, batch_size, num_workers, data_mode):
    train_index_list = list(pd.read_csv(train_index_path)['indexes'])
    val_index_list = list(pd.read_csv(val_index_path)['indexes'])
    img_info = pd.read_csv(img_info_path)

    if data_mode == 'self_evaluated':
        train_index_list = train_index_list[:206]

    train_dataset = SkinDataset(source_dir, img_info, train_index_list, shape, is_test=False)
    val_dataset = SkinDataset(source_dir, img_info, val_index_list, shape, is_test=True)

    train_dataloader = _build_loader(train_dataset, batch_size, num_workers, shuffle=True)
    val_dataloader = _build_loader(val_dataset, batch_size, num_workers, shuffle=True)
    return train_dataloader, val_dataloader
