"""
run_slicer.py
-------------
각 subject × 3축(axial/coronal/sagittal) × 4모달리티(T1/T1ce/T2/FLAIR) 별로
개별 PNG(img_size × img_size)를 저장하고,
subject × modality 단위의 train/test CSV를 생성한다.

이미지 구성 (Loader에서):
    T1  : [axial_T1.png | coronal_T1.png | sagittal_T1.png]  → [3, H, W]
    T1ce: [axial_T1ce.png | ...]                              → [3, H, W]
    T2  : ...
    FLAIR: ...
    ⇒ 4행/subject, 각 행의 3채널 = 3축

Output PNG:
    slices_out/EGD-0085/
        axial_T1.png  axial_T1ce.png  axial_T2.png  axial_FLAIR.png
        coronal_T1.png  ...
        sagittal_T1.png  ...
    (모두 img_size × img_size grayscale)

CSV 컬럼 (한 행 = subject, 12개 PNG + 라벨):
    subject_id,
    T1_axial, T1_coronal, T1_sagittal,
    T1ce_axial, T1ce_coronal, T1ce_sagittal,
    T2_axial, T2_coronal, T2_sagittal,
    FLAIR_axial, FLAIR_coronal, FLAIR_sagittal,
    idh, codel, grade, age, sex

Usage:
    python run_slicer.py \\
        --csv       datasets/EGD_slice.csv \\
        --label_csv datasets/EGD_Final.csv \\
        --out       slices_out \\
        --csv_out   datasets \\
        [--img_size 224] [--workers 4] [--seed 42]

GPU 실행:
    CUDA_VISIBLE_DEVICES=3 python train.py       # GPU 3번을 cuda:0 으로 사용
    HAEDALConfig(device="cuda:3")                # CUDA_VISIBLE_DEVICES 없이 직접 지정
"""

import argparse
import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from HAEDAL_Slicer import find_max_tumor_slice, load_nifti

AXES       = {"axial": 2, "coronal": 1, "sagittal": 0}
MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]
GRADE_MAP  = {"2": 0, "3": 1, "4": 2}

def normalize_volume(v: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(v.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    fg = v > 0
    if np.any(fg):
        vals = v[fg]
        p1, p99 = np.percentile(vals, [1, 99])
        v = np.clip(v, p1, p99)
        if p99 - p1 > 1e-8:
            v = (v - p1) / (p99 - p1)
        else:
            v = np.zeros_like(v, dtype=np.float32)
    else:
        v = np.zeros_like(v, dtype=np.float32)
    return np.clip(v, 0, 1)

def compute_brain_bbox(volumes: dict, eps: float = 1e-6, margin: int = 1):
    fg = None
    for v in volumes.values():
        cur = v > eps
        fg = cur if fg is None else (fg | cur)

    coords = np.where(fg)
    if len(coords[0]) == 0:
        shape = next(iter(volumes.values())).shape
        return (0, shape[0], 0, shape[1], 0, shape[2])

    x1, x2 = coords[0].min(), coords[0].max()
    y1, y2 = coords[1].min(), coords[1].max()
    z1, z2 = coords[2].min(), coords[2].max()

    shape = next(iter(volumes.values())).shape
    x1 = max(0, x1 - margin)
    x2 = min(shape[0], x2 + 1 + margin)
    y1 = max(0, y1 - margin)
    y2 = min(shape[1], y2 + 1 + margin)
    z1 = max(0, z1 - margin)
    z2 = min(shape[2], z2 + 1 + margin)

    return x1, x2, y1, y2, z1, z2

def crop_3d(vol: np.ndarray, bbox):
    x1, x2, y1, y2, z1, z2 = bbox
    return vol[x1:x2, y1:y2, z1:z2]

# ── 단일 subject 처리 ──────────────────────────────────────────────────────────

def process_subject(row: dict, out_dir: str, img_size: int, base_dir: str):
    """
    3축 × 4모달리티 = 12장 PNG 저장 (각 img_size × img_size grayscale).
    Returns (subject_id, status, note, saved)
      saved = { axis: { mod: relative_path } }  ← base_dir 기준 상대경로
    """
    sid = row["subject_id"]
    subj_out = Path(out_dir) / sid
    subj_out.mkdir(parents=True, exist_ok=True)

    try:
        volumes = {
            "T1":    normalize_volume(load_nifti(row["t1"])),
            "T1ce":  normalize_volume(load_nifti(row["t1ce"])),
            "T2":    normalize_volume(load_nifti(row["t2"])),
            "FLAIR": normalize_volume(load_nifti(row["flair"])),
        }
        mask = (load_nifti(row["mask"]) > 0).astype(np.uint8)

        bbox = compute_brain_bbox(volumes, eps=1e-6, margin=5)
        volumes = {k: crop_3d(v, bbox) for k, v in volumes.items()}
        mask = crop_3d(mask, bbox)

        saved = {}
        for axis_name, axis_idx in AXES.items():
            n = find_max_tumor_slice(mask, axis_idx)
            saved[axis_name] = {}
            for mod in MODALITIES:
                sl = np.take(volumes[mod], n, axis=axis_idx).astype(np.float32)
                pil = Image.fromarray(
                    (sl * 255).clip(0, 255).astype(np.uint8), mode="L"
                ).resize((img_size, img_size), Image.BILINEAR)
                png_path = subj_out / f"{axis_name}_{mod}.png"
                pil.save(png_path)
                # base_dir 기준 상대경로로 저장
                saved[axis_name][mod] = str(png_path.resolve().relative_to(base_dir))

        return sid, "ok", "", saved

    except Exception as e:
        return sid, "error", str(e), {}


# ── 라벨 CSV 생성 ──────────────────────────────────────────────────────────────

def build_labeled_csv(
    ok_paths: dict,   # {sid: {axis: {mod: path}}}
    label_csv: str,
    csv_out: str,
    seed: int,
):
    # 라벨 로드
    labels = {}
    with open(label_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            sid = r["subject_id"].strip()
            labels[sid] = {
                "idh":   int(r["IDH"]),
                "codel": int(r["1p19q"]),
                "grade": GRADE_MAP.get(r["Grade"].strip(), -1),
                "age":   r["Age"].strip(),
                "sex":   r["Sex"].strip(),
            }

    # subject 단위로 70/30 분할
    subjects = sorted(ok_paths.keys())
    random.seed(seed)
    random.shuffle(subjects)
    n_train   = round(len(subjects) * 0.7)
    train_set = set(subjects[:n_train])
    test_set  = set(subjects[n_train:])

    no_label = [s for s in subjects if s not in labels]
    if no_label:
        print(f"  [WARN] 라벨 없는 subject {len(no_label)}건 제외: {no_label[:5]}")

    # 컬럼: subject_id + 12 PNG (mod_axis) + 라벨
    png_cols = [f"{mod}_{axis}" for mod in MODALITIES for axis in ("axial", "coronal", "sagittal")]
    fields   = ["subject_id"] + png_cols + ["idh", "codel", "grade", "age", "sex"]

    Path(csv_out).mkdir(parents=True, exist_ok=True)
    for fname, sid_set in [("PDGM_train.csv", train_set), ("PDGM_test.csv", test_set)]:
        rows = []
        for sid in sorted(sid_set):
            if sid not in labels:
                continue
            lb  = labels[sid]
            row = {"subject_id": sid}
            for mod in MODALITIES:
                for axis in ("axial", "coronal", "sagittal"):
                    row[f"{mod}_{axis}"] = ok_paths[sid][axis][mod]
            row.update({"idh": lb["idh"], "codel": lb["codel"], "grade": lb["grade"],
                        "age": lb["age"], "sex": lb["sex"]})
            rows.append(row)
        p = Path(csv_out) / fname
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  {fname}: {len(rows)}명 (× 4모달리티 = Loader에서 {len(rows)*4}샘플) → {p}")

    print(f"  총 subject {len(subjects)}명  (train {len(train_set)} / test {len(test_set)})")


# ── 메인 ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="datasets/EGD_slice.csv")
    parser.add_argument("--label_csv", default="datasets/EGD_Final.csv")
    parser.add_argument("--out",       default="slices_out")
    parser.add_argument("--csv_out",   default="datasets")
    parser.add_argument("--img_size",  type=int, default=224,
                        help="각 PNG 저장 크기 (= DINOv2 입력 해상도)")
    parser.add_argument("--workers",   type=int, default=1)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    base_dir = SCRIPT_DIR  # 상대경로 기준 = 이 스크립트가 있는 HAEDAL 루트

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Subjects: {len(rows)}  |  img_size: {args.img_size}px  |  workers: {args.workers}")
    print(f"PNG out  → {os.path.abspath(args.out)}")
    print(f"base_dir → {base_dir}  (CSV 내 상대경로 기준)\n")

    log_rows = []
    ok_paths = {}
    ok = err = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(process_subject, row, args.out, args.img_size, str(base_dir)): row["subject_id"]
            for row in rows
        }
        for i, fut in enumerate(as_completed(futs), 1):
            sid, status, note, saved = fut.result()
            log_rows.append((sid, status, note))
            if status == "ok":
                ok += 1
                ok_paths[sid] = saved
            else:
                err += 1
                print(f"  [ERROR] {sid}: {note}")
            if i % 20 == 0 or i == len(rows):
                print(f"  [{i:3d}/{len(rows)}] ok={ok} err={err}")

    log_path = Path(args.out) / "slicer_log.tsv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        f.write("subject_id\tstatus\tnote\n")
        for r in sorted(log_rows):
            f.write("\t".join(r) + "\n")
    print(f"\nSlicing done.  ok={ok}  error={err}")
    print(f"Log → {log_path}")

    if os.path.exists(args.label_csv):
        print(f"\n── CSV 생성 ({args.label_csv}) ──")
        build_labeled_csv(ok_paths, args.label_csv, args.csv_out, args.seed)
    else:
        print(f"\n[SKIP] label_csv 없음: {args.label_csv}")


if __name__ == "__main__":
    main()
