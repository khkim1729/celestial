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

Dataset 아이템 (한 subject → 4샘플):
    modality T1   : [axial_T1.png, coronal_T1.png, sagittal_T1.png]   → [3, H, W]
    modality T1ce : [axial_T1ce.png, ...]                              → [3, H, W]
    modality T2   : ...
    modality FLAIR: ...

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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from HAEDAL_Config import HAEDALConfig
from HAEDAL_ClinicalClassifier import encode_clinical

MODALITIES = ("T1", "T1ce", "T2", "FLAIR")
AXES       = ("axial", "coronal", "sagittal")


class HAEDALDataset(Dataset):
    def __init__(self, csv_path: str, cfg: HAEDALConfig, augment: bool = False):
        self.cfg      = cfg
        self.augment  = augment
        self.base_dir = cfg.base_dir   # 상대경로 기준 디렉토리 (빈 문자열이면 그대로)
        # items: (record_dict, modality)  — 4 items per subject row
        self.items    = self._load_csv(csv_path)

    def _abs(self, path: str) -> str:
        """상대경로인 경우 base_dir 기준 절대경로로 변환."""
        if self.base_dir and not os.path.isabs(path):
            return os.path.join(self.base_dir, path)
        return path

    def _load_csv(self, path: str) -> List[Tuple[Dict, str]]:
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
                # 12 PNG paths keyed as "{mod}_{axis}"
                **{f"{mod}_{axis}": r[f"{mod}_{axis}"]
                   for mod in MODALITIES for axis in AXES},
            }
            for mod in MODALITIES:
                items.append((rec, mod))
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec, mod = self.items[idx]
        img_size  = self.cfg.img_size

        # 3채널 = 3축 (axial / coronal / sagittal)
        channels = []
        for axis in AXES:
            pil = Image.open(self._abs(rec[f"{mod}_{axis}"])).convert("L")
            if pil.size != (img_size, img_size):
                pil = pil.resize((img_size, img_size), Image.BILINEAR)
            channels.append(np.array(pil, dtype=np.float32) / 255.0)

        img = torch.from_numpy(np.stack(channels, axis=0))   # [3, H, W]
        if self.augment:
            img = self._aug(img)

        return {
            "image":      img,
            "idh":        torch.tensor(rec["idh"],       dtype=torch.long),
            "codel":      torch.tensor(rec["codel"],     dtype=torch.long),
            "grade":      torch.tensor(rec["grade"],     dtype=torch.long),
            "age_group":  torch.tensor(rec["age_group"], dtype=torch.long),
            "sex_bin":    torch.tensor(rec["sex_bin"],   dtype=torch.long),
            "subject_id": rec["subject_id"],
            "modality":   mod,
        }

    @staticmethod
    def _aug(img: torch.Tensor) -> torch.Tensor:
        import random
        if random.random() < 0.5: img = TF.hflip(img)
        if random.random() < 0.5: img = TF.vflip(img)
        if random.random() < 0.3: img = TF.rotate(img, float(random.randint(-15, 15)))
        return img


def make_loader(csv_path: str, cfg: HAEDALConfig, split: str = "train") -> DataLoader:
    is_train = split == "train"
    ds = HAEDALDataset(csv_path, cfg, augment=is_train)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=is_train,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=is_train,
    )
