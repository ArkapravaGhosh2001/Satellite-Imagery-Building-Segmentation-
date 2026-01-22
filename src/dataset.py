import random
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

class SpaceNetPatchDataset(Dataset):
    def __init__(self, image_paths, mask_paths, patch_size=256, empty_prob=0.15, max_retries=15):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.empty_prob = empty_prob
        self.max_retries = max_retries

    def __len__(self):
        return len(self.image_paths) * 100  # virtual length

    def _read_pair(self, i):
        with rasterio.open(self.image_paths[i]) as src:
            img = src.read([1, 2, 3])  # (3,H,W)

        with rasterio.open(self.mask_paths[i]) as src:
            mask = src.read(1)         # (H,W) values 0/1

        return img, mask

    def __getitem__(self, idx):
        i = idx % len(self.image_paths)
        img, mask = self._read_pair(i)

        _, H, W = img.shape
        ps = self.patch_size

        for _ in range(self.max_retries):
            x = random.randint(0, W - ps)
            y = random.randint(0, H - ps)

            img_patch = img[:, y:y+ps, x:x+ps]
            mask_patch = mask[y:y+ps, x:x+ps]

            if mask_patch.sum() > 0:
                break
            if random.random() < self.empty_prob:
                break

        img_patch = img_patch.astype(np.float32) / 255.0
        img_patch = torch.from_numpy(img_patch)         # float32 [3,ps,ps]
        mask_patch = torch.from_numpy(mask_patch).long()# int64 [ps,ps]

        return img_patch, mask_patch
