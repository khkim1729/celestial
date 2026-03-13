"""
HAEDAL_PlotClinical.py
-----------------------
실제 환자의 임상 feature (age_group / sex) 기준 예측 확률 시각화.

저장 파일: {save_dir}/{subject_id}_clinical.png

Figure layout (1 row × 3 cols):
  [0,0] IDH      예측 확률
  [0,1] 1p/19q   예측 확률
  [0,2] Grade    예측 확률
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path

TASKS = {
    "idh":   {"names": ["IDH-wt",    "IDH-mut"],
              "colors": ["#E07B7B", "#7BA7E0"]},
    "codel": {"names": ["Non-codel", "Codeleted"],
              "colors": ["#A6D854", "#FFD92F"]},
    "grade": {"names": ["Grade2",    "Grade3",   "Grade4"],
              "colors": ["#66C2A5", "#FC8D62", "#8DA0CB"]},
}
TASK_TITLES = {"idh": "IDH", "codel": "1p/19q", "grade": "Grade"}
TASK_KEYS   = ["idh", "codel", "grade"]


def _infer(model, img_tensor, age_group, sex):
    model.eval()
    with torch.no_grad():
        idh_p, codel_p, grade_p = model(img_tensor, age_group, sex)
    return {
        "idh":   F.softmax(idh_p,   dim=1)[0].cpu().numpy(),
        "codel": F.softmax(codel_p, dim=1)[0].cpu().numpy(),
        "grade": F.softmax(grade_p, dim=1)[0].cpu().numpy(),
    }


def _plot_probs(ax, task_key, probs, pred_cls, true_cls, title):
    meta   = TASKS[task_key]
    x      = np.arange(len(meta["names"]))
    bars   = ax.bar(x, probs, 0.5, color=meta["colors"],
                    edgecolor="white", linewidth=0.5, alpha=0.85)

    bars[pred_cls].set_edgecolor("black")
    bars[pred_cls].set_linewidth(2.0)

    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.01,
                f"{p:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(meta["names"], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Probability", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    if true_cls >= 0:
        ax.text(0.01, 1.08, f"GT: {meta['names'][true_cls]}",
                transform=ax.transAxes, fontsize=8, color="#555555", va="top")


def generate_clinical_plot(
    model,
    img_tensor,      # [1, 4, 3, H, W]
    age_group,       # [1] long tensor
    sex,             # [1] long tensor
    pred_classes,    # {"idh": int, "codel": int, "grade": int}
    true_classes,    # {"idh": int, "codel": int, "grade": int}  (-1 = unknown)
    save_dir,
    subject_id: str = "",
):
    ag_str = "Old(≥45)" if age_group.item() == 1 else "Young(<45)"
    sx_str = "Male"     if sex.item()       == 1 else "Female"

    result = _infer(model, img_tensor, age_group, sex)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    suptitle  = f"{subject_id}  |  Age: {ag_str}  Sex: {sx_str}" if subject_id \
                else f"Age: {ag_str}  Sex: {sx_str}"
    fig.suptitle(suptitle, fontsize=11, y=1.02)

    for col, task in enumerate(TASK_KEYS):
        pred_cls  = pred_classes[task]
        true_cls  = true_classes.get(task, -1)
        pred_name = TASKS[task]["names"][pred_cls]
        true_name = TASKS[task]["names"][true_cls] if true_cls >= 0 else "?"
        mark      = "✓" if pred_cls == true_cls else "✗"
        title     = f"{TASK_TITLES[task]}  pred={pred_name}  gt={true_name} {mark}"

        _plot_probs(axes[col], task, result[task], pred_cls, true_cls, title)

    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{subject_id}_clinical.png" if subject_id else "clinical.png"
    plt.savefig(save_dir / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)