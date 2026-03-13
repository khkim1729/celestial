# HAEDAL — Hierarchical Glioma Multi-Task Classifier

## 모듈 구성

| 파일 | 역할 |
|---|---|
| `HAEDAL_Classifier.py` | DINOv2 기반 계층적 분류기 (IDH→1p19q→Grade) |
| `HAEDAL_ClinicalClassifier.py` | 임상 데이터 통합 분류기 (`clinical_dim=0`이면 임상 생략) |
| `HAEDAL_Loss.py` | 계층적 손실 + WHO 제약 페널티 (결측 라벨 -1 지원) |
| `HAEDAL_Config.py` | 하이퍼파라미터 데이터클래스 (`base_dir`, `mask_brain` 포함) |
| `HAEDAL_Slicer.py` | 3D MRI → 2D 슬라이스 추출 (축별 최대종양 슬라이스) |
| `HAEDAL_Loader.py` | PNG 기반 Dataset / DataLoader (상대경로 자동 해석, 뇌 마스킹 지원) |
| `HAEDAL_Metrics.py` | 다중 평가지표 계산 + TSV 저장 |
| `HAEDAL_Trainer.py` | 학습 루프 (조기 종료 포함) |
| `HAEDAL_Tester.py` | 테스트 평가 (TTA / TTT / Grad-CAM / ClinicalPlot) |
| `HAEDAL_GradCAM.py` | 축별(axial/coronal/sagittal) 독립 Grad-CAM 시각화 |
| `HAEDAL_TTA.py` | Test Time Augmentation (예측 평균) |
| `HAEDAL_TTT.py` | Test Time Training (엔트로피 최소화) |
| `HAEDAL_PlotClinical.py` | 환자별 임상 피처(나이/성별) 예측 확률 시각화 |
| `HAEDAL_PlotDistribution.py` | 데이터셋 내 Age/Sex × 라벨 분포 시각화 |
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
| `idh` | 0=wild-type, 1=mutant, **-1=unknown** |
| `codel` | 0=non-codeleted, 1=codeleted, **-1=unknown** |
| `grade` | 0=Grade2, 1=Grade3, 2=Grade4, **-1=unknown** |
| `age` | float (예: 42.0) |
| `sex` | "M" 또는 "F" |

> **결측 라벨(-1)**: 손실 계산 시 해당 태스크를 자동 제외. 예측·평가는 정상 진행.
> **EGD Grade 변환:** 원본 2/3/4 → HAEDAL 0/1/2
> **상대경로 → 절대경로 변환**은 `run_trainer.py` / `run_tester.py` 가 자동 처리 (`HAEDALConfig.base_dir`)

### 3단계 — Dataset (`HAEDAL_Loader`)

CSV 1행(subject) → **1 샘플** `[4, 3, H, W]`

```
image [4, 3, 224, 224]:
  dim 0 : modality  (T1 / T1ce / T2 / FLAIR)
  dim 1 : axis채널  (axial / coronal / sagittal)  ← DINOv2 RGB 입력으로 활용
```

예시 (T1 modality):
```
ch0 = axial_T1.png    ┐
ch1 = coronal_T1.png  ├→ stack → [3, 224, 224]  → DINOv2 입력
ch2 = sagittal_T1.png ┘
```

**뇌 마스킹** (`--mask_brain`): near-zero 픽셀을 0으로 만들어 뇌 외부 배경 제거.
임계값 = `max_pixel × 2%`

---

## 모델 아키텍처 (`HAEDAL_ClinicalClassifier`)

```
image [B, 4, 3, 224, 224]
    │
    ├─ view [B*4, 3, 224, 224] ──→ DINOv2 ViT-g/14 ──→ [B*4, 1536]
    │                                                        │
    └─────────────────────────── mean(dim=1) ───────────── [B, 1536] = img_feat
                                                              │
age_group, sex_bin  ──→  ClinicalEncoder  ──→  [B, clinical_dim]
                                                              │
                                              fused = concat [B, 1536 + clinical_dim]
                                                              │
                                              idh_head   → idh_logits   [B, 2]
                                                  ↓ + idh_context
                                              codel_head → codel_logits [B, 2]
                                                  ↓ + genetic_context
                                              grade_head → grade_logits [B, 3]
```

> `clinical_dim=0` 이면 ClinicalEncoder를 생략하고 `fused = img_feat [B, 1536]` 그대로 사용.

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

**결측 라벨(-1) 처리:**
- `target.clamp(min=0)` 후 CE 계산 → index error 방지
- `valid_mask = (target >= 0).float()` 로 손실 마스킹
- `loss.sum() / n_valid.clamp(min=1)` 로 정규화
- `is_idh_wrong`: 라벨 있는 샘플에만 계산

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
    --clinical_dim   64 \    # 0이면 ClinicalEncoder 생략
    --mask_brain \           # 뇌 외부 픽셀 마스킹
    --device         cuda
```

`--no_freeze` : DINOv2 백본 동결 해제
`--no_amp`    : AMP(mixed precision) 비활성화

### Step 2 — 테스트 (`run_tester.py`)

```bash
# 기본 실행
CUDA_VISIBLE_DEVICES=3 python run_tester.py \
    --checkpoint output/exp01/checkpoints/best.pt

# 전체 옵션
CUDA_VISIBLE_DEVICES=3 python run_tester.py \
    --test_csv          datasets/EGD_test.csv \
    --checkpoint        output/exp01/checkpoints/best.pt \
    --experiment        exp01 \
    --output_dir        output \
    --batch_size        16 \
    --clinical_dim      64 \     # 학습 시와 동일하게 설정
    --mask_brain \               # 뇌 외부 픽셀 마스킹
    --tta \                      # Test Time Augmentation
    --tta_n             8 \      # augmentation 횟수
    --ttt \                      # Test Time Training
    --ttt_steps         10 \     # 엔트로피 최소화 스텝 수
    --ttt_lr            1e-4 \   # TTT learning rate
    --gradcam \                  # Grad-CAM 시각화 생성
    --gradcam_samples   20 \     # Grad-CAM 생성 최대 샘플 수 (기본: 전체)
    --clinical_plot \            # 환자별 임상 영향 시각화
    --clinical_samples  50 \     # clinical_plot 최대 샘플 수
    --device            cuda
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

## Test Time Augmentation / Training

### TTA (`--tta`)

원본 + (n_aug-1)개의 augmented 버전을 모델에 통과시켜 softmax 확률을 평균.
백본 가중치 변경 없음. `n_aug=1`이면 TTA 없이 단순 예측과 동일.

```bash
--tta --tta_n 8
```

### TTT (`--ttt`, Tent-style)

각 샘플에 대해 헤드 파라미터만 엔트로피 최소화 방향으로 임시 업데이트한 뒤 예측.
예측 후 원래 가중치로 복원 (샘플 간 독립).

```bash
--ttt --ttt_steps 10 --ttt_lr 1e-4
```

> 업데이트 대상: `base.backbone`을 제외한 파라미터 (`freeze_backbone` 여부와 무관하게 백본 제외)

---

## Grad-CAM 시각화 (`--gradcam`)

**저장 위치:** `output/{exp}/gradcam/{subject_id}_{mod}.png`

**Figure layout (4 rows × 3 cols, 모달리티당 1 PNG):**

| 행 | 내용 |
|---|---|
| Row 0 | 원본 슬라이스 (axial / coronal / sagittal) |
| Row 1 | IDH Grad-CAM 오버레이 |
| Row 2 | 1p/19q Grad-CAM 오버레이 |
| Row 3 | Grade Grad-CAM 오버레이 |

**축별 독립 CAM:**
각 열(column)은 해당 축 채널만 활성화한 modified input으로 `blocks[-1]` Grad-CAM 수행.
3축이 같은 CAM을 공유하지 않고 각 축의 고유 공간 중요도를 시각화.

---

## 임상 피처 시각화

### 환자별 예측 확률 (`--clinical_plot`)

**저장 위치:** `output/{exp}/clinical/{subject_id}_clinical.png`

환자 실제 나이/성별에서의 모델 예측 확률을 3개 태스크(IDH / 1p19q / Grade)별 막대 그래프로 표시.
예측 클래스는 검은 테두리, 정답 라벨은 텍스트 표시.

### 데이터셋 분포 (`HAEDAL_PlotDistribution`)

Age group(Young/Old) × 라벨, Sex × 라벨 분포를 stacked bar로 시각화.

---

## 출력 파일 구조

```
output/{exp}/
    checkpoints/
        best.pt              # 최적 체크포인트
        latest.pt            # 최신 체크포인트
    history.json             # 전체 에포크 히스토리
    history_metrics.tsv      # 에포크별 지표 (train + val)
    test_metrics_{ts}.tsv    # 테스트 지표
    test_metrics_{ts}.json
    test_results_{ts}.csv    # 샘플별 예측 결과
    gradcam/                 # Grad-CAM PNG (--gradcam 실행 시)
        {subject_id}_T1.png
        {subject_id}_T1ce.png
        {subject_id}_T2.png
        {subject_id}_FLAIR.png
    clinical/                # 임상 영향 PNG (--clinical_plot 실행 시)
        {subject_id}_clinical.png
```

---

## Config 주요 파라미터 (`HAEDAL_Config`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `train_csv` | `"data/train.csv"` | 학습 CSV 경로 |
| `val_csv` | `"data/val.csv"` | 검증 CSV 경로 |
| `img_size` | `224` | 입력 해상도 (DINOv2 기본값) |
| `freeze_backbone` | `True` | DINOv2 백본 동결 여부 |
| `clinical_dim` | `64` | ClinicalEncoder 출력 차원 (0 = 임상 생략) |
| `age_cutoff` | `45` | Young/Old 구분 기준 나이 |
| `epochs` | `50` | 학습 에포크 수 |
| `batch_size` | `8` | 배치 크기 |
| `lr` | `1e-4` | 학습률 |
| `scheduler` | `"cosine"` | LR 스케줄러 (cosine / step / none) |
| `warmup_epochs` | `5` | Warmup 에포크 수 |
| `w_idh` | `3.0` | IDH 손실 가중치 |
| `w_codel` | `1.0` | 1p/19q 손실 가중치 |
| `w_grade` | `1.0` | Grade 손실 가중치 |
| `early_stop_patience` | `15` | 조기 종료 인내 에포크 (0 = 비활성화) |
| `early_stop_min_delta` | `1e-4` | 조기 종료 최소 개선량 |
| `mask_brain` | `False` | 뇌 외부 near-zero 픽셀 마스킹 |
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
opencv-python
matplotlib
```

설치:

```bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
pip install numpy Pillow nibabel scikit-learn opencv-python matplotlib
```
