# HAEDAL — Hierarchical Glioma Multi-Task Classifier

## 모듈 구성

| 파일 | 역할 |
|---|---|
| `HAEDAL_Classifier.py` | DINOv2 기반 계층적 분류기 (IDH→1p19q→Grade) |
| `HAEDAL_ClinicalClassifier.py` | 임상 데이터 통합 분류기 (ClinicalEncoder + 계층적 헤드) |
| `HAEDAL_Loss.py` | 계층적 손실 + WHO 제약 페널티 |
| `HAEDAL_Config.py` | 하이퍼파라미터 데이터클래스 (`base_dir` 포함) |
| `HAEDAL_Slicer.py` | 3D MRI → 2D 슬라이스 추출 (축별 최대종양 슬라이스) |
| `HAEDAL_Loader.py` | PNG 기반 Dataset / DataLoader (상대경로 자동 해석) |
| `HAEDAL_Metrics.py` | 다중 평가지표 계산 + TSV 저장 |
| `HAEDAL_Trainer.py` | 학습 루프 |
| `HAEDAL_Tester.py` | 테스트 평가 |
| `run_slicer.py` | NIfTI → PNG 전처리 + train/test CSV 생성 |
| `run_trainer.py` | 학습 실행 스크립트 (경로 자동 절대화) |
| `run_tester.py` | 테스트 실행 스크립트 (경로 자동 절대화) |

---

## 데이터 파이프라인

### 1단계 — NIfTI 전처리 (`run_slicer.py`)

```bash
python run_slicer.py \
    --csv       datasets/EGD_slice.csv \   # NIfTI 경로 목록
    --label_csv datasets/EGD_Final.csv \   # 라벨 (IDH/1p19q/Grade/Age/Sex)
    --out       slices_out \               # PNG 저장 디렉토리
    --csv_out   datasets \                 # train/test CSV 저장 위치
    --img_size  224 \                      # PNG 저장 해상도
    --workers   4 \                        # 병렬 처리 수
    --seed      42                         # train/test 분할 시드 (70/30)
```

각 subject당 **12장** PNG 저장 (모두 224×224 grayscale):

```
slices_out/{subject_id}/
    axial_T1.png    axial_T1ce.png    axial_T2.png    axial_FLAIR.png
    coronal_T1.png  coronal_T1ce.png  coronal_T2.png  coronal_FLAIR.png
    sagittal_T1.png sagittal_T1ce.png sagittal_T2.png sagittal_FLAIR.png
```

슬라이스 선택 기준: 세그멘테이션 마스크 기준 **종양 면적이 가장 큰 슬라이스** (축별).

완료 후 자동 생성:
```
slices_out/slicer_log.tsv          # 처리 결과 로그
datasets/EGD_train.csv             # subject 70%
datasets/EGD_test.csv              # subject 30%
```

### 2단계 — CSV 포맷

**한 행 = 1 subject, 컬럼 18개 (PNG 경로는 HAEDAL 루트 기준 상대경로):**

```
subject_id,
T1_axial, T1_coronal, T1_sagittal,
T1ce_axial, T1ce_coronal, T1ce_sagittal,
T2_axial,  T2_coronal,  T2_sagittal,
FLAIR_axial, FLAIR_coronal, FLAIR_sagittal,
idh, codel, grade, age, sex
```

| 컬럼 | 값 |
|---|---|
| `T1_axial` 등 12개 | PNG **상대경로** (HAEDAL 루트 기준, 예: `slices_out/EGD-0086/axial_T1.png`) |
| `idh` | 0=wild-type, 1=mutant, -1=unknown |
| `codel` | 0=non-codeleted, 1=codeleted, -1=unknown |
| `grade` | 0=Grade2, 1=Grade3, 2=Grade4, -1=unknown |
| `age` | float (예: 42.0) |
| `sex` | "M" 또는 "F" |

> **EGD Grade 변환:** 원본 2/3/4 → HAEDAL 0/1/2
> **상대경로 → 절대경로 변환**은 `run_trainer.py` / `run_tester.py` 가 자동 처리 (`HAEDALConfig.base_dir`)

### 3단계 — Dataset (`HAEDAL_Loader`)

CSV 1행(subject) → **4 아이템** (T1 / T1ce / T2 / FLAIR)

각 아이템 이미지 구성:

```
modality T1:
  ch0 = axial_T1.png    ┐
  ch1 = coronal_T1.png  ├→ stack → [3, 224, 224]  (DINOv2 RGB 입력)
  ch2 = sagittal_T1.png ┘

3채널 = 3축 (axial / coronal / sagittal)
```

---

## 모델 아키텍처 (`HAEDAL_ClinicalClassifier`)

```
image [B, 3, 224, 224]  →  DINOv2 ViT-g/14  →  img_feat  [B, 1536]
age_group, sex_bin      →  ClinicalEncoder  →  clin_feat [B, clinical_dim]
                                                           ↓
                                              fused = concat [B, 1536+clinical_dim]
                                                           ↓
                                              idh_head   → idh_logits   [B, 2]
                                                  ↓ + idh_context
                                              codel_head → codel_logits [B, 2]
                                                  ↓ + genetic_context
                                              grade_head → grade_logits [B, 3]
```

`ClinicalEncoder`: Linear(2→64) → ReLU → Dropout(0.1) → Linear(64→`clinical_dim`) → ReLU

임상 피처 변환:

| 원본 | 변환 | 설명 |
|---|---|---|
| `age` (float) | `age_group` 0/1 | < 45 → Young(0), ≥ 45 → Old(1) |
| `sex` ("M"/"F") | `sex_bin` 0/1 | F→0, M→1 |

---

## 손실 함수 (`HAEDAL_Loss`)

```
L = w_idh × L_idh
  + w_codel × L_codel × dependency_mask
  + w_grade × L_grade × dependency_mask
  + penalty

dependency_mask : IDH 틀린 샘플의 하위 태스크 손실 0.2배 억제
penalty         : IDH-wt인데 codeleted로 예측 시 +2.0 페널티 (WHO 규칙 위반)
기본 가중치      : w_idh=3.0, w_codel=1.0, w_grade=1.0
```

---

## 평가 지표 (`HAEDAL_Metrics`)

태스크별(IDH / 1p19q / Grade)로 계산:

| 지표 | 설명 |
|---|---|
| `acc` | Accuracy |
| `bal_acc` | Balanced Accuracy |
| `f1_macro` | F1 (macro average) |
| `f1_weighted` | F1 (weighted average) |
| `f1_{class}` | 클래스별 F1 |
| `precision` | Precision |
| `recall` | Recall |
| `auc` | ROC-AUC (이진: IDH, 1p/19q) |
| `auc_ovr` | ROC-AUC OvR (Grade, 다중클래스) |
| `auc_ovo` | ROC-AUC OvO (Grade, 다중클래스) |
| `mcc` | Matthews Correlation Coefficient |
| `kappa` | Cohen's Kappa |
| `confusion_matrix` | 혼동 행렬 |

**Overall score** = (mean_acc + IDH_AUC) / 2

학습/테스트 완료 시 자동 저장:

```
output/{exp}/history_metrics.tsv     # 에포크별 전체 지표 (train + val)
output/{exp}/test_metrics_{ts}.tsv   # 테스트 전체 지표
output/{exp}/test_metrics_{ts}.json
output/{exp}/test_results_{ts}.csv   # 샘플별 예측 결과
```

---

## 실행 방법

### Step 0 — 전처리 (최초 1회)

```bash
python run_slicer.py \
    --csv       datasets/EGD_slice.csv \
    --label_csv datasets/EGD_Final.csv \
    --out       slices_out \
    --csv_out   datasets \
    --img_size  224 \
    --workers   4 \
    --seed      42
```

### Step 1 — 학습 (`run_trainer.py`)

```bash
# GPU 3번 사용
CUDA_VISIBLE_DEVICES=3 python run_trainer.py \
    --experiment exp01 \
    --epochs     150 \
    --batch_size 16

# 전체 옵션
CUDA_VISIBLE_DEVICES=3 python run_trainer.py \
    --train_csv      datasets/EGD_train.csv \
    --val_csv        datasets/EGD_test.csv \
    --experiment     exp01 \
    --output_dir     output \
    --epochs         150 \
    --batch_size     16 \
    --lr             1e-4 \
    --scheduler      cosine \
    --warmup         5 \
    --num_workers    4 \
    --w_idh          3.0 \
    --w_codel        1.0 \
    --w_grade        1.0 \
    --device         cuda          # CUDA_VISIBLE_DEVICES 로 GPU 지정
                                   # 또는 --device cuda:3 으로 직접 지정
```

`--no_freeze` : DINOv2 백본 동결 해제
`--no_amp`    : AMP(mixed precision) 비활성화

### Step 2 — 테스트 (`run_tester.py`)

```bash
# GPU 3번 사용
CUDA_VISIBLE_DEVICES=3 python run_tester.py \
    --checkpoint output/exp01/checkpoints/best.pt

# 전체 옵션
CUDA_VISIBLE_DEVICES=3 python run_tester.py \
    --test_csv   datasets/EGD_test.csv \
    --checkpoint output/exp01/checkpoints/best.pt \
    --experiment exp01 \
    --output_dir output \
    --batch_size 16 \
    --device     cuda
```

### 다른 서버에서 실행 시

CSV 내 PNG 경로는 **상대경로**로 저장되어 있어 서버 간 이동이 가능하다.
프로젝트 루트 경로가 달라진 경우 `--base_dir` 로 명시한다.

```bash
CUDA_VISIBLE_DEVICES=3 python run_trainer.py \
    --base_dir /new/server/path/HAEDAL \
    --experiment exp01

CUDA_VISIBLE_DEVICES=3 python run_tester.py \
    --base_dir   /new/server/path/HAEDAL \
    --checkpoint /new/server/path/HAEDAL/output/exp01/checkpoints/best.pt
```

### 경로 해석 흐름

```
run_slicer.py   → CSV에 상대경로 저장  (HAEDAL 루트 기준)
                  예: slices_out/EGD-0086/axial_T1.png

run_trainer.py  → base_dir = 스크립트 위치 (자동) 또는 --base_dir
run_tester.py   →   CSV/output 절대화 → HAEDALConfig.base_dir 에 전달

HAEDAL_Loader   → cfg.base_dir + 상대경로 → 절대경로로 PNG 로드
```

---

## Config 주요 파라미터 (`HAEDAL_Config`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `train_csv` | `"data/train.csv"` | 학습 CSV 경로 |
| `val_csv` | `"data/val.csv"` | 검증 CSV 경로 |
| `img_size` | `224` | 입력 해상도 (DINOv2 기본값) |
| `freeze_backbone` | `True` | DINOv2 백본 동결 여부 |
| `clinical_dim` | `64` | ClinicalEncoder 출력 차원 |
| `age_cutoff` | `45` | Young/Old 구분 기준 나이 |
| `epochs` | `50` | 학습 에포크 수 |
| `batch_size` | `8` | 배치 크기 |
| `lr` | `1e-4` | 학습률 |
| `scheduler` | `"cosine"` | LR 스케줄러 (cosine / step / none) |
| `w_idh` | `3.0` | IDH 손실 가중치 |
| `device` | `"cuda"` | 디바이스 (`"cuda"` / `"cuda:3"` 등) |
| `base_dir` | `""` | PNG 상대경로 기준 디렉토리 (`run_trainer/tester` 가 자동 설정) |

---

## 의존성

```
torch==2.5.1+cu124
torchvision==0.20.1+cu124
numpy
Pillow>=10.0
nibabel
scikit-learn>=1.3
```

설치:

```bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
pip install numpy Pillow nibabel scikit-learn
```
