from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class HAEDALConfig:
    # Data
    train_csv:   str   = "data/train.csv"
    val_csv:     str   = "data/val.csv"
    test_csv:    str   = "data/test.csv"
    modalities:  Tuple = ("T1", "T1ce", "T2", "FLAIR")
    axes:        Tuple = ("axial", "coronal", "sagittal")
    img_size:    int   = 224   # DINOv2 input resolution

    # Model
    num_grades:       int   = 3      # WHO Grade 2/3/4
    freeze_backbone:  bool  = True
    clinical_dim:     int   = 64     # ClinicalEncoder output dimension
    age_cutoff:       int   = 45     # Young: age < cutoff, Old: age >= cutoff

    # Training
    epochs:        int   = 50
    batch_size:    int   = 8
    num_workers:   int   = 4
    lr:            float = 1e-4
    weight_decay:  float = 1e-5
    scheduler:     str   = "cosine"  # cosine | step | none
    warmup_epochs: int   = 5
    amp:           bool  = True
    seed:          int   = 42

    # Loss weights (mirrors HierarchicalGliomaLoss defaults)
    w_idh:   float = 3.0
    w_codel: float = 1.0
    w_grade: float = 1.0

    # Output
    output_dir:       str = "output"
    experiment_name:  str = "haedal"
    save_best_metric: str = "score"   # key from metrics dict

    # Device
    device: str = "cuda"

    # Early stopping (0 = 비활성화)
    early_stop_patience:  int   = 15
    early_stop_min_delta: float = 1e-4

    # Preprocessing
    mask_brain: bool = False   # True: 뇌 외부 near-zero 픽셀 마스킹

    # Paths — CSV 내 상대경로의 기준 디렉토리 (빈 문자열이면 변환 없음)
    base_dir: str = ""
