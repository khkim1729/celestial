"""
HAEDAL_GradCAM.py
-----------------
Grad-CAM visualization for HierarchicalGliomaClassifierWithClinical.

입력 형식: [1, 4, 3, H, W]  (1 subject, 4 modalities, 3 axes per modality)

Per-axis Grad-CAM:
  각 축(axial/coronal/sagittal)에 대해 독립적으로 CAM 계산.
  해당 축 채널만 활성화하고 나머지 두 채널을 0으로 만든 modified input으로
  blocks[-1] Grad-CAM 수행 → 축별 공간 중요도 분리.

출력: 모달리티당 1 PNG
  {save_dir}/{subject_id}_{mod}.png
  Figure layout  (4 rows × 3 cols):
    Row 0 : 원본 슬라이스 (axial / coronal / sagittal)
    Row 1 : IDH   Grad-CAM 오버레이  (각 열 = 해당 축 전용 CAM)
    Row 2 : 1p/19q Grad-CAM 오버레이
    Row 3 : Grade Grad-CAM 오버레이

Install:  pip install opencv-python matplotlib
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path

MODALITIES  = ("T1",   "T1ce",  "T2",    "FLAIR")
TASK_KEYS   = ("idh",  "codel", "grade")
TASK_LABELS = ("IDH",  "1p/19q","Grade")
AXIS_NAMES  = ("axial","coronal","sagittal")

_VIT_GRID   = 16    # 224 / 14 = 16 패치
_INPUT_SIZE = 224


def _make_axis_input(img_tensor: torch.Tensor, mod_idx: int, axis_idx: int) -> torch.Tensor:
    """
    img_tensor: [1, 4, 3, H, W]
    modality mod_idx 에서 axis_idx 채널만 남기고 나머지를 0으로 설정한 복사본 반환.
    다른 modality는 그대로 유지.
    """
    x = img_tensor.clone()
    axis_slice = img_tensor[0, mod_idx, axis_idx]          # [H, W]
    x[0, mod_idx] = 0.0
    x[0, mod_idx, axis_idx] = axis_slice
    return x


def _gradcam_vit(
    model:      nn.Module,
    img_tensor: torch.Tensor,  # [1, 4, 3, H, W]
    age_group:  torch.Tensor,  # [1] long
    sex:        torch.Tensor,  # [1] long
    task_idx:   int,
    pred_cls:   int,
    mod_idx:    int,           # 0=T1 / 1=T1ce / 2=T2 / 3=FLAIR
) -> np.ndarray:
    """
    지정된 modality 에 대한 Grad-CAM 히트맵을 반환한다.

    model.forward 내부에서 [1, 4, 3, H, W] → [4, 3, H, W] 로 reshape 되므로
    blocks[-1] 입력은 [4, 257, 1536].  mod_idx 번째 슬라이스가 해당 모달리티.

    Returns
    -------
    cam : np.ndarray  shape [224, 224], dtype float32, range [0, 1]
    """
    buf = {}

    def _fwd(module, inp, out):
        buf["act"] = inp[0]   # [4, 257, 1536] — gradient graph 유지

    def _bwd(module, grad_in, grad_out):
        buf["grad"] = grad_in[0].detach() if grad_in[0] is not None else None

    last_block = model.base.backbone.blocks[-1]
    h_fwd = last_block.register_forward_hook(_fwd)
    h_bwd = last_block.register_full_backward_hook(_bwd)

    try:
        model.eval()
        model.zero_grad()

        x       = img_tensor.detach().requires_grad_(True)
        outputs = model(x, age_group, sex)
        score   = outputs[task_idx][0, pred_cls]
        score.backward()

        grad = buf.get("grad")
        if grad is None:
            # fallback: input image gradient saliency for this modality
            cam_np = x.grad[0, mod_idx].abs().mean(0).cpu().numpy().astype(np.float32)
        else:
            act_m  = buf["act"][mod_idx, 1:, :].detach()  # [256, 1536]
            grad_m = grad[mod_idx, 1:, :]                  # [256, 1536]

            weights = grad_m.mean(dim=0)                   # [1536]
            cam     = torch.einsum("nd,d->n", act_m, weights)  # [256]
            cam     = cam.reshape(_VIT_GRID, _VIT_GRID)
            cam     = torch.relu(cam)
            cam_np  = cam.cpu().numpy().astype(np.float32)

        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        cam_np = cv2.resize(cam_np, (_INPUT_SIZE, _INPUT_SIZE),
                            interpolation=cv2.INTER_LINEAR)
        return cam_np

    finally:
        h_fwd.remove()
        h_bwd.remove()
        model.zero_grad()


def _overlay(gray_img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """gray [H,W] + CAM [H,W] → RGB 오버레이 [H,W,3]."""
    rgb  = np.stack([gray_img] * 3, axis=-1)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(0.5 * rgb + 0.5 * heat, 0, 1)


def generate_gradcam(
    model,
    img_tensor,     # [1, 4, 3, H, W]
    age_group,      # [1] long tensor
    sex,            # [1] long tensor
    pred_classes,   # {"idh": int, "codel": int, "grade": int}
    true_classes,   # {"idh": int, "codel": int, "grade": int}  (-1 = unknown)
    save_dir,       # Path or str
    subject_id: str = "",
):
    """
    4 modality 별로 Grad-CAM 시각화를 저장한다.

    저장 파일: {save_dir}/{subject_id}_{mod}.png  (T1 / T1ce / T2 / FLAIR)

    Figure layout  (4 rows × 3 cols):
      Row 0 : 원본 슬라이스  (axial / coronal / sagittal)
      Row 1 : IDH   Grad-CAM 오버레이
      Row 2 : 1p/19q Grad-CAM 오버레이
      Row 3 : Grade Grad-CAM 오버레이
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # [1, 4, 3, H, W] → [4, 3, H, W] numpy
    img_np = img_tensor.squeeze(0).cpu().float().numpy()   # [4, 3, H, W]
    img_np = np.clip(img_np, 0.0, 1.0)

    for mod_idx, mod_name in enumerate(MODALITIES):
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        title = f"{subject_id}  {mod_name}" if subject_id else mod_name
        fig.suptitle(title, fontsize=10, y=1.005)

        # ── Row 0: 원본 슬라이스 ────────────────────────────────────────
        for ch_idx, axis_name in enumerate(AXIS_NAMES):
            ax = axes[0, ch_idx]
            ax.imshow(img_np[mod_idx, ch_idx], cmap="gray", vmin=0, vmax=1)
            ax.set_title(axis_name, fontsize=10)
            ax.axis("off")
        axes[0, 0].set_ylabel("Original", fontsize=9)

        # ── Rows 1–3: task 별 Grad-CAM (축별 독립 CAM) ─────────────────────────
        for task_idx, (task_key, task_label) in enumerate(
            zip(TASK_KEYS, TASK_LABELS)
        ):
            pred_cls = pred_classes[task_key]
            true_cls = true_classes.get(task_key, -1)

            row_label = f"{task_label}\npred={pred_cls}"
            if true_cls >= 0:
                mark = "✓" if pred_cls == true_cls else "✗"
                row_label += f"  gt={true_cls} {mark}"

            for ch_idx in range(3):
                # 해당 축만 활성화한 modified input으로 axis-specific CAM 계산
                x_axis = _make_axis_input(img_tensor, mod_idx, ch_idx)
                cam = _gradcam_vit(
                    model      = model,
                    img_tensor = x_axis,
                    age_group  = age_group,
                    sex        = sex,
                    task_idx   = task_idx,
                    pred_cls   = pred_cls,
                    mod_idx    = mod_idx,
                )

                # 뇌 외부 CAM 억제: ViT 내부 활성화는 입력 마스킹과 무관하게
                # 배경 패치에서도 값이 새어나오므로 사후에 brain mask 적용.
                orig = img_np[mod_idx, ch_idx]   # [H, W]
                peak = orig.max()
                if peak > 1.0 / 255:
                    brain_mask = (orig > peak * 0.02).astype(np.float32)
                    cam = cam * brain_mask

                ax = axes[task_idx + 1, ch_idx]
                vis = _overlay(img_np[mod_idx, ch_idx], cam)
                ax.imshow(vis)
                ax.axis("off")

            axes[task_idx + 1, 0].set_ylabel(row_label, fontsize=9)

        plt.tight_layout()
        fname = f"{subject_id}_{mod_name}.png" if subject_id else f"{mod_name}.png"
        plt.savefig(save_dir / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
