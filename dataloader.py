import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


class MMCDataset(Dataset):
    def __init__(
        self,
        meta_csv,
        index_csv,
        image_dir,
        image_size,
        fundus_col,
        oct_col,
        label_col,
        class_names,
        is_train=False,
    ):
        self.image_dir = Path(image_dir)
        self.image_size = (image_size, image_size)
        self.fundus_col = fundus_col
        self.oct_col = oct_col
        self.label_col = label_col
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.rows = self._read_rows(meta_csv)
        self.indexes = [int(row["indexes"]) for row in self._read_rows(index_csv)]
        self.is_train = is_train

    @staticmethod
    def _read_rows(path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def __len__(self):
        return len(self.indexes)

    def _load_image(self, relative_path):
        image = Image.open(self.image_dir / relative_path).convert("RGB")
        return image.resize(self.image_size)

    def _augment_pair(self, fundus, oct_img):
        if random.random() < 0.5:
            fundus = ImageOps.mirror(fundus)
            oct_img = ImageOps.mirror(oct_img)
        if random.random() < 0.5:
            fundus = ImageOps.flip(fundus)
            oct_img = ImageOps.flip(oct_img)
        return fundus, oct_img

    @staticmethod
    def _to_tensor(image):
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        data = data.view(image.size[1], image.size[0], 3)
        return data.permute(2, 0, 1).float().div(255.0)

    def __getitem__(self, item):
        row = self.rows[self.indexes[item]]
        fundus = self._load_image(row[self.fundus_col])
        oct_img = self._load_image(row[self.oct_col])
        if self.is_train:
            fundus, oct_img = self._augment_pair(fundus, oct_img)

        label = self.class_to_idx[row[self.label_col]]
        return self._to_tensor(fundus), self._to_tensor(oct_img), torch.tensor(label, dtype=torch.long)


def build_dataloader(args, index_csv, class_names, is_train):
    dataset = MMCDataset(
        meta_csv=args.meta_csv,
        index_csv=index_csv,
        image_dir=args.image_dir,
        image_size=args.image_size,
        fundus_col=args.fundus_col,
        oct_col=args.oct_col,
        label_col=args.label_col,
        class_names=class_names,
        is_train=is_train,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
