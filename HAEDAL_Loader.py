"""
HAEDAL_Loader.py
----------------
PNG 기반 Dataset / DataLoader.

CSV 포맷 (한 행 = subject, 총 12 PNG + 라벨):
    subject_id,
    T1_axial, T1_coronal, T1_sagittal,
    T1ce_axial, T1ce_coronal, T1ce_sagittal,
    T2_axial,  T2_coronal,  T2_sagittal,
    FLAIR_axial, FLAIR_coronal, FLAIR_sagittal,
    idh, codel, grade, age, sex

Dataset 아이템 (한 subject → 1 샘플):
    image : [4, 3, H, W]
            dim 0 : modality  (T1 / T1ce / T2 / FLAIR)
            dim 1 : axis      (axial / coronal / sagittal)

Labels (int):
    idh   : 0=wild-type  1=mutant            (-1 = unknown)
    codel : 0=non-codel  1=codeleted         (-1 = unknown)
    grade : 0=Grade2     1=Grade3  2=Grade4  (-1 = unknown)

Clinical:
    age : float → age_group 0=Young(<45) / 1=Old(>=45)
    sex : "M"/"F" → sex_bin 0=F / 1=M
"""

import csv
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from HAEDAL_Config import HAEDALConfig
from HAEDAL_ClinicalClassifier import encode_clinical

MODALITIES = ("T1", "T1ce", "T2", "FLAIR")
AXES       = ("axial", "coronal", "sagittal")


def _brain_mask(img: np.ndarray) -> np.ndarray:
    """
    img: [H, W] float32 [0, 1]
    near-zero 배경 픽셀을 0으로 만드는 binary mask 반환.
    임계값 = max × 2% (이미지가 거의 검으면 전부 1 반환).
    """
    peak = img.max()
    if peak < 1.0 / 255:          # 거의 검은 이미지
        return np.ones_like(img)
    thresh = peak * 0.02
    return (img > thresh).astype(np.float32)


class HAEDALDataset(Dataset):
    def __init__(self, csv_path: str, cfg: HAEDALConfig, augment: bool = False):
        self.cfg      = cfg
        self.augment  = augment
        self.base_dir = cfg.base_dir
        self.items    = self._load_csv(csv_path)

    def _abs(self, path: str) -> str:
        if self.base_dir and not os.path.isabs(path):
            return os.path.join(self.base_dir, path)
        return path

    def _load_csv(self, path: str) -> List[Dict]:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        items = []
        for r in rows:
            ag, sx = encode_clinical(r.get("age", 0), r.get("sex", "F"), self.cfg.age_cutoff)
            rec = {
                "subject_id": r["subject_id"],
                "idh":        int(r.get("idh",   -1)),
                "codel":      int(r.get("codel", -1)),
                "grade":      int(r.get("grade", -1)),
                "age_group":  ag,
                "sex_bin":    sx,
                **{f"{mod}_{axis}": r[f"{mod}_{axis}"]
                   for mod in MODALITIES for axis in AXES},
            }
            items.append(rec)   # 1 subject → 1 item
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec      = self.items[idx]
        img_size = self.cfg.img_size

        # 4 modalities, 각각 3채널(3축) → [4, 3, H, W]
        modality_imgs = []
        for mod in MODALITIES:
            channels = []
            for axis in AXES:
                pil = Image.open(self._abs(rec[f"{mod}_{axis}"])).convert("L")
                if pil.size != (img_size, img_size):
                    pil = pil.resize((img_size, img_size), Image.BILINEAR)
                arr = np.array(pil, dtype=np.float32) / 255.0
                if self.cfg.mask_brain:
                    arr = arr * _brain_mask(arr)
                channels.append(arr)
            modality_imgs.append(np.stack(channels, axis=0))  # [3, H, W]

        imgs = torch.from_numpy(np.stack(modality_imgs, axis=0))  # [4, 3, H, W]

        if self.augment:
            imgs = self._aug(imgs)

        return {
            "image":      imgs,
            "idh":        torch.tensor(rec["idh"],       dtype=torch.long),
            "codel":      torch.tensor(rec["codel"],     dtype=torch.long),
            "grade":      torch.tensor(rec["grade"],     dtype=torch.long),
            "age_group":  torch.tensor(rec["age_group"], dtype=torch.long),
            "sex_bin":    torch.tensor(rec["sex_bin"],   dtype=torch.long),
            "subject_id": rec["subject_id"],
        }

    @staticmethod
    def _aug(imgs: torch.Tensor) -> torch.Tensor:
        """4 modality 에 동일한 랜덤 변환을 적용해 공간 정합성 유지."""
        do_hflip  = random.random() < 0.5
        do_vflip  = random.random() < 0.5
        do_rotate = random.random() < 0.3
        angle     = float(random.randint(-15, 15))

        augmented = []
        for i in range(imgs.shape[0]):   # iterate over modalities
            img = imgs[i]
            if do_hflip:  img = TF.hflip(img)
            if do_vflip:  img = TF.vflip(img)
            if do_rotate: img = TF.rotate(img, angle)
            augmented.append(img)
        return torch.stack(augmented, dim=0)


def make_loader(csv_path: str, cfg: HAEDALConfig, split: str = "train") -> DataLoader:
    is_train = split == "train"
    ds = HAEDALDataset(csv_path, cfg, augment=is_train)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=is_train,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=is_train,
    )