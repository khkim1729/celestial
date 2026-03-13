"""
HAEDAL_TTA.py
-------------
Test Time Augmentation.

동일 샘플에 랜덤 augmentation 을 n_aug 회 적용해 예측 확률을 평균.
원본(i=0) + 랜덤 변환(i=1..n_aug-1) 의 softmax 평균을 반환.
"""

import torch
import torch.nn.functional as F

from HAEDAL_Loader import HAEDALDataset   # _aug() 재사용


@torch.no_grad()
def tta_infer(
    model,
    imgs:    torch.Tensor,   # [B, 4, 3, H, W]
    age_g:   torch.Tensor,   # [B]
    sex_b:   torch.Tensor,   # [B]
    n_aug:   int = 8,
):
    """
    Returns
    -------
    idh_prob   : [B, 2]
    codel_prob : [B, 2]
    grade_prob : [B, 3]
    — 모두 softmax 확률, n_aug 회 평균값
    """
    model.eval()
    B = imgs.shape[0]

    sum_idh   = torch.zeros(B, 2, device=imgs.device)
    sum_codel = torch.zeros(B, 2, device=imgs.device)
    sum_grade = torch.zeros(B, 3, device=imgs.device)

    for i in range(n_aug):
        if i == 0:
            aug = imgs
        else:
            # 각 샘플에 독립적으로 augmentation 적용
            aug = torch.stack(
                [HAEDALDataset._aug(imgs[b]) for b in range(B)], dim=0
            )

        idh_p, codel_p, grade_p = model(aug, age_g, sex_b)
        sum_idh   += F.softmax(idh_p,   dim=1)
        sum_codel += F.softmax(codel_p, dim=1)
        sum_grade += F.softmax(grade_p, dim=1)

    return sum_idh / n_aug, sum_codel / n_aug, sum_grade / n_aug