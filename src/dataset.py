"""RGB tensors"""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PixelItemDataset(Dataset):
    def __init__(self, manifest="data/processed/manifest.csv", split="train", image_size=32):
        self.manifest = Path(manifest)
        self.image_size = image_size
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val, or all")
        with self.manifest.open(encoding="utf-8-sig", newline="") as stream:
            self.rows = [row for row in csv.DictReader(stream)
                         if row["split"] in {"train", "val"} and (split == "all" or row["split"] == split)]
        if not self.rows:
            raise ValueError(f"No images for split {split!r} in {self.manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path = self.manifest.parent / self.rows[index]["training_path"]
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (self.image_size, self.image_size):
                raise ValueError(f"Unexpected mode/size: {path}: {image.mode} {image.size}")
            array = np.array(image, dtype=np.float32, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous().div_(127.5).sub_(1.0)
