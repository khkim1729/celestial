"""
run_tester.py
-------------
HAEDAL 테스트 평가 스크립트.
CSV 내 상대경로를 스크립트 위치 기준 절대경로로 자동 변환한다.

Usage:
    # GPU 3번 사용
    CUDA_VISIBLE_DEVICES=3 python run_tester.py \\
        --checkpoint output/haedal_exp01/checkpoints/best.pt

    # 옵션 지정
    CUDA_VISIBLE_DEVICES=3 python run_tester.py \\
        --test_csv   datasets/EGD_test.csv \\
        --checkpoint output/haedal_exp01/checkpoints/best.pt \\
        --experiment haedal_exp01 \\
        --output_dir output

    # 다른 서버에서 실행 시
    CUDA_VISIBLE_DEVICES=3 python run_tester.py \\
        --checkpoint /path/to/best.pt \\
        --base_dir   /new/server/path/HAEDAL
"""

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()


def resolve(path: str, base: Path) -> str:
    p = Path(path)
    return str((base / p).resolve()) if not p.is_absolute() else str(p)


def main():
    parser = argparse.ArgumentParser(description="HAEDAL Tester")
    parser.add_argument("--test_csv",    default="datasets/EGD_test.csv")
    parser.add_argument("--checkpoint",  required=True,
                        help="체크포인트 경로 (best.pt 등)")
    parser.add_argument("--base_dir",    default=str(SCRIPT_DIR),
                        help="CSV 내 PNG 상대경로 기준 디렉토리 (기본: 스크립트 위치)")
    parser.add_argument("--experiment",  default="haedal_exp01",
                        dest="experiment_name")
    parser.add_argument("--output_dir",  default="output")
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--gradcam",         action="store_true",
                        help="Grad-CAM 시각화 생성 (requires: pip install grad-cam)")
    parser.add_argument("--gradcam_samples", type=int, default=None,
                        help="Grad-CAM 생성 최대 샘플 수 (기본: 전체)")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()

    test_csv    = resolve(args.test_csv,    base)
    checkpoint  = resolve(args.checkpoint,  base)
    output_dir  = resolve(args.output_dir,  base)

    print(f"[run_tester] base_dir   : {base}")
    print(f"[run_tester] test_csv   : {test_csv}")
    print(f"[run_tester] checkpoint : {checkpoint}")
    print(f"[run_tester] output_dir : {output_dir}")
    print(f"[run_tester] device     : {args.device}")
    print()

    from HAEDAL_Config import HAEDALConfig
    from HAEDAL_Tester import HAEDALTester

    cfg = HAEDALConfig(
        test_csv        = test_csv,
        output_dir      = output_dir,
        experiment_name = args.experiment_name,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        device          = args.device,
        base_dir        = str(base),
    )

    tester = HAEDALTester(cfg, checkpoint=checkpoint)
    tester.evaluate(test_csv)

    if args.gradcam:
        tester.gradcam(test_csv, max_samples=args.gradcam_samples)


if __name__ == "__main__":
    main()
