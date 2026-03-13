"""
run_trainer.py
--------------
HAEDAL 학습 실행 스크립트.
CSV 내 상대경로를 스크립트 위치 기준 절대경로로 자동 변환한다.

Usage:
    # GPU 3번 사용
    CUDA_VISIBLE_DEVICES=3 python run_trainer.py

    # 옵션 지정
    CUDA_VISIBLE_DEVICES=3 python run_trainer.py \\
        --train_csv      datasets/EGD_train.csv \\
        --val_csv        datasets/EGD_test.csv \\
        --experiment     exp01 \\
        --epochs         150 \\
        --batch_size     16 \\
        --lr             1e-4 \\
        --output_dir     output

    # 다른 서버에서 실행 시 base_dir 명시 (CSV 내 상대경로 기준 디렉토리)
    CUDA_VISIBLE_DEVICES=3 python run_trainer.py \\
        --base_dir /new/server/path/HAEDAL
"""

import argparse
import os
from pathlib import Path

# 이 스크립트가 위치한 디렉토리 = HAEDAL 루트
SCRIPT_DIR = Path(__file__).parent.resolve()


def resolve(path: str, base: Path) -> str:
    """상대경로면 base 기준 절대경로로 변환, 절대경로면 그대로."""
    p = Path(path)
    return str((base / p).resolve()) if not p.is_absolute() else str(p)


def main():
    parser = argparse.ArgumentParser(description="HAEDAL Trainer")
    parser.add_argument("--train_csv",   default="datasets/EGD_train.csv")
    parser.add_argument("--val_csv",     default="datasets/EGD_test.csv")
    parser.add_argument("--base_dir",    default=str(SCRIPT_DIR),
                        help="CSV 내 PNG 상대경로 기준 디렉토리 (기본: 스크립트 위치)")
    parser.add_argument("--experiment",  default="haedal_exp01",
                        dest="experiment_name")
    parser.add_argument("--output_dir",  default="output")
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-5)
    parser.add_argument("--scheduler",   default="cosine",
                        choices=["cosine", "step", "none"])
    parser.add_argument("--warmup",      type=int,   default=5,
                        dest="warmup_epochs")
    parser.add_argument("--no_freeze",   action="store_true",
                        help="DINOv2 백본 동결 해제")
    parser.add_argument("--no_amp",      action="store_true",
                        help="AMP(mixed precision) 비활성화")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      default="cuda",
                        help="'cuda' (CUDA_VISIBLE_DEVICES로 GPU 지정) 또는 'cuda:N'")
    parser.add_argument("--w_idh",       type=float, default=3.0)
    parser.add_argument("--w_codel",     type=float, default=1.0)
    parser.add_argument("--w_grade",     type=float, default=1.0)
    parser.add_argument("--save_best",   default="score",
                        dest="save_best_metric")
    parser.add_argument("--clinical_dim", type=int, default=64,
                        help="ClinicalEncoder 출력 차원 (0 = clinical 미사용)")
    parser.add_argument("--mask_brain",  action="store_true",
                        help="뇌 외부 near-zero 픽셀 마스킹")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()

    # 상대경로 → 절대경로 변환
    train_csv  = resolve(args.train_csv,  base)
    val_csv    = resolve(args.val_csv,    base)
    output_dir = resolve(args.output_dir, base)

    print(f"[run_trainer] base_dir   : {base}")
    print(f"[run_trainer] train_csv  : {train_csv}")
    print(f"[run_trainer] val_csv    : {val_csv}")
    print(f"[run_trainer] output_dir : {output_dir}")
    print(f"[run_trainer] device     : {args.device}")
    print()

    # import here so CUDA_VISIBLE_DEVICES is respected before torch init
    from HAEDAL_Config import HAEDALConfig
    from HAEDAL_Trainer import HAEDALTrainer

    cfg = HAEDALConfig(
        train_csv        = train_csv,
        val_csv          = val_csv,
        output_dir       = output_dir,
        experiment_name  = args.experiment_name,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        num_workers      = args.num_workers,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        scheduler        = args.scheduler,
        warmup_epochs    = args.warmup_epochs,
        freeze_backbone  = not args.no_freeze,
        amp              = not args.no_amp,
        seed             = args.seed,
        device           = args.device,
        w_idh            = args.w_idh,
        w_codel          = args.w_codel,
        w_grade          = args.w_grade,
        save_best_metric = args.save_best_metric,
        base_dir         = str(base),
        clinical_dim     = args.clinical_dim,
        mask_brain       = args.mask_brain,
    )

    HAEDALTrainer(cfg).train()


if __name__ == "__main__":
    main()
