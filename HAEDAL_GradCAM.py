"""
HAEDAL_GradCAM.py
-----------------
Grad-CAM visualization for HierarchicalGliomaClassifierWithClinical.

DINOv2 ViT-g/14 는 pytorch-grad-cam 의 hook API 와 호환되지 않으므로
forward/backward hook 을 직접 등록해 Grad-CAM 을 계산한다.

Target layer : model.base.backbone.norm  (blocks 이후 최종 LayerNorm)
  - 출력: [B, 257, 1536]  (1 CLS + 16×16 패치)
  - 패치 토큰 [1:, :] → reshape [16, 16, 1536] → GAP → ReLU → 224×224 upsample

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

TASK_KEYS   = ("idh",   "codel",  "grade")
TASK_LABELS = ("IDH",   "1p/19q", "Grade")
AXIS_NAMES  = ("axial", "coronal", "sagittal")

_VIT_GRID   = 16   # 224 / 14 = 16 패치
_INPUT_SIZE = 224


def _gradcam_vit(
    model: nn.Module,
    img_tensor: torch.Tensor,   # [1, 3, 224, 224]
    age_group:  torch.Tensor,   # [1] long
    sex:        torch.Tensor,   # [1] long
    task_idx:   int,
    pred_cls:   int,
) -> np.ndarray:
    """
    DINOv2 의 마지막 transformer block 입력에 hook 을 등록해 Grad-CAM 히트맵을 계산한다.

    Target: model.base.backbone.blocks[-1] (NestedTensorBlock)
      - forward hook  : 블록 입력  [B, 257, 1536] 저장
      - backward hook : 블록 입력에 대한 gradient [B, 257, 1536] 저장
      patch 위치(1:)에서 gradient 가 attention 을 통해 non-zero 로 흐른다.

    Returns
    -------
    cam : np.ndarray  shape [224, 224], dtype float32, range [0, 1]
    """
    buf = {}

    def _fwd(module, inp, out):
        # inp[0]: 블록 입력 [B, 257, 1536]  (gradient graph 유지를 위해 detach 하지 않음)
        buf["act"] = inp[0]

    def _bwd(module, grad_in, grad_out):
        # grad_in[0]: d(loss)/d(block_input) [B, 257, 1536]
        buf["grad"] = grad_in[0].detach() if grad_in[0] is not None else None

    last_block = model.base.backbone.blocks[-1]
    h_fwd = last_block.register_forward_hook(_fwd)
    h_bwd = last_block.register_full_backward_hook(_bwd)

    try:
        model.eval()
        model.zero_grad()

        x = img_tensor.detach().requires_grad_(True)
        outputs = model(x, age_group, sex)
        score   = outputs[task_idx][0, pred_cls]
        score.backward()

        grad = buf.get("grad")
        if grad is None:
            # fallback: input image gradient saliency
            cam_np = x.grad[0].abs().mean(0).cpu().numpy().astype(np.float32)
        else:
            act  = buf["act"][0, 1:, :].detach()   # [256, 1536]  patch tokens
            grad = grad[0, 1:, :]                   # [256, 1536]

            # Grad-CAM: 채널별 gradient GAP → 가중합
            weights = grad.mean(dim=0)              # [1536]
            cam = torch.einsum("nd,d->n", act, weights)   # [256]
            cam = cam.reshape(_VIT_GRID, _VIT_GRID)        # [16, 16]
            cam = torch.relu(cam)
            cam_np = cam.cpu().numpy().astype(np.float32)

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
    """
    단일 채널 gray 이미지 [H, W] + CAM [H, W] → RGB 오버레이 [H, W, 3].
    """
    rgb  = np.stack([gray_img] * 3, axis=-1)               # [H, W, 3]
    heat = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )                                                        # [H, W, 3] BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out  = 0.5 * rgb + 0.5 * heat
    return np.clip(out, 0, 1)


def generate_gradcam(
    model,
    img_tensor,     # [1, 3, 224, 224]
    age_group,      # [1] long tensor
    sex,            # [1] long tensor
    pred_classes,   # {"idh": int, "codel": int, "grade": int}
    true_classes,   # {"idh": int, "codel": int, "grade": int}  (-1 = unknown)
    save_path,
    title: str = "",
):
    """
    3 tasks × 3 axes Grad-CAM 시각화를 save_path 에 저장.

    Figure layout  (4 rows × 3 cols):
      Row 0 : 원본 슬라이스  (axial / coronal / sagittal)
      Row 1 : IDH   Grad-CAM 오버레이
      Row 2 : 1p19q Grad-CAM 오버레이
      Row 3 : Grade Grad-CAM 오버레이
    """
    model.eval()

    # [1, 3, 224, 224] → [224, 224, 3] float32 in [0, 1]
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    img_np = np.clip(img_np, 0.0, 1.0)

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    if title:
        fig.suptitle(title, fontsize=10, y=1.005)

    # ── Row 0: 원본 슬라이스 ─────────────────────────────────────────────
    for ch_idx, axis_name in enumerate(AXIS_NAMES):
        ax = axes[0, ch_idx]
        ax.imshow(img_np[:, :, ch_idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(axis_name, fontsize=10)
        ax.axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=9)

    # ── Rows 1–3: task 별 Grad-CAM ──────────────────────────────────────
    for task_idx, (task_key, task_label) in enumerate(
        zip(TASK_KEYS, TASK_LABELS)
    ):
        pred_cls = pred_classes[task_key]
        true_cls = true_classes.get(task_key, -1)

        cam = _gradcam_vit(
            model     = model,
            img_tensor= img_tensor,
            age_group = age_group,
            sex       = sex,
            task_idx  = task_idx,
            pred_cls  = pred_cls,
        )

        row_label = f"{task_label}\npred={pred_cls}"
        if true_cls >= 0:
            mark = "✓" if pred_cls == true_cls else "✗"
            row_label += f"  gt={true_cls} {mark}"

        for ch_idx in range(3):
            ax = axes[task_idx + 1, ch_idx]
            vis = _overlay(img_np[:, :, ch_idx], cam)
            ax.imshow(vis)
            ax.axis("off")

        axes[task_idx + 1, 0].set_ylabel(row_label, fontsize=9)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
