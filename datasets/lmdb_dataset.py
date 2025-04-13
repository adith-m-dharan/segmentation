import lmdb
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class LmdbSegmentationDataset(Dataset):
    def __init__(self, image_lmdb_path, mask_lmdb_path, transform=None, target_transform=None):
        self.image_env = lmdb.open(image_lmdb_path, readonly=True, lock=False)
        self.mask_env = lmdb.open(mask_lmdb_path, readonly=True, lock=False)

        with self.image_env.begin() as txn:
            self.keys = [key for key, _ in txn.cursor()]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]

        with self.image_env.begin() as img_txn, self.mask_env.begin() as mask_txn:
            img_buf = img_txn.get(key)
            mask_buf = mask_txn.get(key)

        if img_buf is None or mask_buf is None:
            raise RuntimeError(f"Missing key {key.decode()} in LMDB.")

        # Decode using IMREAD_UNCHANGED for speed and minimal conversion
        img = cv2.imdecode(np.frombuffer(img_buf, np.uint8), cv2.IMREAD_UNCHANGED)
        mask = cv2.imdecode(np.frombuffer(mask_buf, np.uint8), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise RuntimeError(f"Corrupted entry for key: {key.decode()}")

        # Ensure correct channel ordering (BGR → RGB) if image has 3 channels
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Optional early cast to save memory
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)

        # Apply transforms
        if self.transform:
            # If transform expects PIL input
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform:
            if not isinstance(mask, Image.Image):
                mask = Image.fromarray(mask)
            mask = self.target_transform(mask)

        # Ensure tensor output
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(np.array(mask)).long()

        return img, mask
