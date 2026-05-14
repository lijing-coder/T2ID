import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class MMCDataset(Dataset):
    def __init__(self, meta_csv, index_csv, image_dir, image_size, fundus_col, oct_col, label_col, class_names):
        self.image_dir = Path(image_dir)
        self.image_size = (image_size, image_size)
        self.fundus_col = fundus_col
        self.oct_col = oct_col
        self.label_col = label_col
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.rows = self._read_rows(meta_csv)
        self.indexes = [int(row["indexes"]) for row in self._read_rows(index_csv)]

    @staticmethod
    def _read_rows(path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def __len__(self):
        return len(self.indexes)

    def _load_image(self, relative_path):
        image = Image.open(self.image_dir / relative_path).convert("RGB")
        image = image.resize(self.image_size)
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        data = data.view(image.size[1], image.size[0], 3)
        return data.permute(2, 0, 1).float().div(255.0)

    def __getitem__(self, item):
        row = self.rows[self.indexes[item]]
        fundus = self._load_image(row[self.fundus_col])
        oct_img = self._load_image(row[self.oct_col])
        label = self.class_to_idx[row[self.label_col]]
        return fundus, oct_img, torch.tensor(label, dtype=torch.long)


def build_eval_dataloader(args, class_names):
    dataset = MMCDataset(
        meta_csv=args.meta_csv,
        index_csv=args.index_csv,
        image_dir=args.image_dir,
        image_size=args.image_size,
        fundus_col=args.fundus_col,
        oct_col=args.oct_col,
        label_col=args.label_col,
        class_names=class_names,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
