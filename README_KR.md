# CELESTIAL

**CELESTIAL: Cascaded Embedding and Latent Encoding via Slicing for Triple-task IDH-centric Autoencoder Learning**

> [English README](./README.md) · [영어 문서](./README.md)

---

## 개요

CELESTIAL은 다중 모달 MRI 기반의 신경교종(glioma) 분자 아형 분류를 위한 계층적 다중 태스크 학습 프레임워크입니다.
2021 WHO CNS 종양 분류 기준의 계층적 의존 구조를 따라, 세 가지 임상적으로 중요한 바이오마커를 동시에 예측합니다:
- **IDH 돌연변이 여부**
- **1p/19q 공동결손(co-deletion)**
- **WHO 등급(Grade)**

주요 구성 요소:
- **3D MRI → 2D 슬라이스 추출**: 종양 면적 최대화 기준으로 축별(axial / coronal / sagittal) 슬라이스 선택
- **DINOv2 ViT-g/14**: 고정(frozen) 비전 백본 (임베딩 차원: 1536)
- **ClinicalEncoder**: 나이·성별 공변량 융합 (`clinical_dim=0`이면 생략 가능)
- **계단식 계층적 분류 헤드**: WHO 제약 페널티 손실 함수 포함
- **결측 라벨(-1) 지원**: IDH / 1p19q / Grade 각각 독립적으로 결측 처리
- **TTA / TTT**: 추론 시 정확도 향상 옵션
- **축별 Grad-CAM**: axial / coronal / sagittal 각 축 독립 시각화

---

## 그림

### 아키텍처 개요

![Fig1 — 시스템 파이프라인](guide/fig1guide.png)

### 슬라이스 전처리

![Fig2 — 다중 모달 MRI 슬라이싱 전략](guide/fig2guide.png)

### Grad-CAM 시각화

![Fig3 — Grad-CAM 어텐션 맵](guide/fig3guide.png)

---

## 모듈 구성

| 파일 | 역할 |
|---|---|
| `HAEDAL_Classifier.py` | DINOv2 기반 계층적 분류기 (IDH → 1p19q → Grade) |
| `HAEDAL_ClinicalClassifier.py` | 임상 데이터 통합 분류기 (`clinical_dim=0`이면 임상 생략) |
| `HAEDAL_Loss.py` | 계층적 손실 + WHO 제약 페널티 (결측 라벨 -1 지원) |
| `HAEDAL_Config.py` | 하이퍼파라미터 데이터클래스 |
| `HAEDAL_Slicer.py` | 3D MRI → 2D 슬라이스 추출 (종양 면적 최대 슬라이스) |
| `HAEDAL_Loader.py` | PNG 기반 Dataset / DataLoader (뇌 마스킹 지원) |
| `HAEDAL_Metrics.py` | 다중 평가지표 계산 + TSV 저장 |
| `HAEDAL_Trainer.py` | 학습 루프 (조기 종료 포함) |
| `HAEDAL_Tester.py` | 테스트 평가 (TTA / TTT / Grad-CAM / ClinicalPlot) |
| `HAEDAL_GradCAM.py` | 축별 독립 Grad-CAM 시각화 |
| `HAEDAL_TTA.py` | Test Time Augmentation |
| `HAEDAL_TTT.py` | Test Time Training (엔트로피 최소화) |
| `HAEDAL_PlotClinical.py` | 환자별 임상 피처 예측 확률 시각화 |
| `HAEDAL_PlotDistribution.py` | 데이터셋 Age/Sex × 라벨 분포 시각화 |
| `run_slicer.py` | NIfTI → PNG 전처리 + train/test CSV 생성 |
| `run_trainer.py` | 학습 실행 스크립트 |
| `run_tester.py` | 테스트 실행 스크립트 |

---

## 데이터 파이프라인

### 0단계 — 전처리 (최초 1회)

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

각 subject당 **12장** PNG 저장 (224×224 grayscale), 3축 × 4 모달리티:

```
slices_out/{subject_id}/
    axial_T1.png    axial_T1ce.png    axial_T2.png    axial_FLAIR.png
    coronal_T1.png  coronal_T1ce.png  coronal_T2.png  coronal_FLAIR.png
    sagittal_T1.png sagittal_T1ce.png sagittal_T2.png sagittal_FLAIR.png
```

슬라이스 선택 기준: 세그멘테이션 마스크 기준 **종양 면적이 가장 큰 슬라이스** (축별).

### 1단계 — 학습

```bash
CUDA_VISIBLE_DEVICES=0 python run_trainer.py \
    --experiment exp01 \
    --epochs     150 \
    --batch_size 16 \
    --clinical_dim 64 \    # 0이면 ClinicalEncoder 생략
    --mask_brain           # 뇌 외부 픽셀 마스킹 (선택)
```

### 2단계 — 테스트

```bash
CUDA_VISIBLE_DEVICES=0 python run_tester.py \
    --checkpoint output/exp01/checkpoints/best.pt \
    --clinical_dim 64 \    # 학습 시와 동일하게 설정
    --tta \                # Test Time Augmentation (선택)
    --gradcam \            # Grad-CAM 시각화 생성 (선택)
    --clinical_plot        # 환자별 임상 영향 시각화 (선택)
```

---

## 모델 아키텍처

```
image [B, 4, 3, 224, 224]
    │
    ├─ view [B*4, 3, 224, 224] ──→ DINOv2 ViT-g/14 ──→ [B*4, 1536]
    │                                                        │
    └─────────────────────────── mean over modalities ──── [B, 1536] = img_feat
                                                              │
age_group, sex_bin  ──→  ClinicalEncoder  ──→  [B, clinical_dim]
                                                              │ (clinical_dim=0이면 생략)
                                              fused [B, 1536 + clinical_dim]
                                                              │
                                              idh_head   → idh_logits   [B, 2]
                                                  ↓ + idh_context [B, 128]
                                              codel_head → codel_logits [B, 2]
                                                  ↓ + genetic_context [B, 256]
                                              grade_head → grade_logits [B, 3]
```

**입력 인코딩:**
- 1 subject → 1 샘플 `[4, 3, 224, 224]`
- dim 0: modality (T1 / T1ce / T2 / FLAIR)
- dim 1: 3축 슬라이스 채널 (axial / coronal / sagittal) → DINOv2 RGB 입력으로 활용

**ClinicalEncoder:**
`Linear(2→64) → ReLU → Dropout(0.1) → Linear(64→clinical_dim) → ReLU`

| 원본 피처 | 인코딩 |
|---|---|
| `age` (float) | 0 = Young (< 45세), 1 = Old (≥ 45세) |
| `sex` (M/F) | 0 = Female, 1 = Male |

---

## 손실 함수

```
L_total = w_idh × L_idh
        + w_codel × L_codel × dependency_mask
        + w_grade × L_grade × dependency_mask
        + penalty

dependency_mask : IDH 예측이 틀린 샘플의 하위 태스크 손실 0.2배 억제
penalty         : IDH-wt인데 1p19q-codeleted로 예측 시 +2.0 (WHO 규칙 위반)
가중치          : w_idh=3.0, w_codel=1.0, w_grade=1.0
```

**결측 라벨(-1) 처리:**
각 태스크(IDH / 1p19q / Grade)에 대해 라벨이 -1인 샘플은 손실 계산에서 자동 제외.
라벨이 있는 샘플만으로 정규화 (`loss.sum() / n_valid`).

---

## 평가 지표

태스크별(IDH / 1p19q / Grade):

| 지표 | 설명 |
|---|---|
| `acc` | 정확도 |
| `bal_acc` | 균형 정확도 |
| `f1_macro` | 매크로 F1 |
| `auc` | ROC-AUC (이진 분류) |
| `auc_ovr` | OvR AUC (Grade, 다중 클래스) |
| `mcc` | Matthews 상관계수 |
| `kappa` | Cohen's Kappa |

**Overall score** = (mean_acc + IDH_AUC) / 2

---

## Test Time Augmentation / Training

### TTA (`--tta`)

원본 + 증강 버전들의 softmax 확률을 평균내어 예측 안정성 향상.

```bash
--tta --tta_n 8    # 8번 augmentation 평균 (기본값)
```

### TTT (`--ttt`, Tent-style)

각 샘플에 대해 헤드 파라미터만 엔트로피 최소화 방향으로 임시 업데이트.
예측 후 원래 가중치 복원 (샘플 간 독립). 백본은 업데이트하지 않음.

```bash
--ttt --ttt_steps 10 --ttt_lr 1e-4
```

---

## Grad-CAM 시각화 (`--gradcam`)

각 subject × 모달리티(T1/T1ce/T2/FLAIR)별로 PNG 1장 저장.

**Figure (4 rows × 3 cols):**

| 행 | 내용 |
|---|---|
| Row 0 | 원본 슬라이스 (axial / coronal / sagittal) |
| Row 1 | IDH Grad-CAM 오버레이 |
| Row 2 | 1p/19q Grad-CAM 오버레이 |
| Row 3 | Grade Grad-CAM 오버레이 |

각 열은 해당 축 채널만 활성화한 modified input으로 독립적으로 Grad-CAM 계산
→ 3축이 같은 CAM을 공유하지 않음.

**저장 위치:** `output/{exp}/gradcam/{subject_id}_{mod}.png`

---

## 임상 영향 시각화 (`--clinical_plot`)

환자의 실제 나이/성별에서의 IDH / 1p19q / Grade 예측 확률을 막대 그래프로 시각화.
예측 클래스는 검은 테두리로 표시, 정답 라벨은 텍스트 표기.

**저장 위치:** `output/{exp}/clinical/{subject_id}_clinical.png`

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
    test_results_{ts}.csv    # 샘플별 예측 결과
    gradcam/                 # Grad-CAM PNG (--gradcam 실행 시)
        {subject_id}_T1.png
        {subject_id}_T1ce.png
        ...
    clinical/                # 임상 영향 PNG (--clinical_plot 실행 시)
        {subject_id}_clinical.png
```

---

## 의존성 패키지

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

```bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
pip install numpy Pillow nibabel scikit-learn opencv-python matplotlib
```

---

## 데이터셋

- **EGD (Erasmus Glioma Database)**: 다중 모달 뇌 MRI (T1, T1ce, T2, FLAIR) + 세그멘테이션 마스크
- 라벨: IDH 상태, 1p/19q 공동결손, WHO 등급, 나이, 성별
- 분할: 70% train / 30% test (계층적 분할, `seed=42`)

---

## 라벨 인코딩

| 라벨 | 값 |
|---|---|
| `idh` | 0=wild-type, 1=mutant, **-1=unknown** |
| `codel` | 0=non-codeleted, 1=codeleted, **-1=unknown** |
| `grade` | 0=Grade2, 1=Grade3, 2=Grade4, **-1=unknown** |
