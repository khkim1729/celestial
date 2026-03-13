"""
HAEDAL_TTT.py
-------------
Test Time Training (Tent-style entropy minimization).

각 샘플에 대해:
  1. backbone 을 제외한 trainable params (heads + ClinicalEncoder) 를 일시적으로 업데이트
  2. 예측 확률의 엔트로피를 최소화하는 방향으로 n_steps 번 gradient step
  3. 최종 예측 수행 후 원래 파라미터 복구

backbone(DINOv2)은 항상 고정 — 업데이트 대상에서 제외.
"""

import copy

import torch
import torch.nn.functional as F


def ttt_infer(
    model,
    imgs:     torch.Tensor,   # [B, 4, 3, H, W]
    age_g:    torch.Tensor,   # [B]
    sex_b:    torch.Tensor,   # [B]
    n_steps:  int   = 10,
    lr:       float = 1e-4,
):
    """
    Returns
    -------
    idh_prob   : [B, 2]
    codel_prob : [B, 2]
    grade_prob : [B, 3]
    — 모두 softmax 확률, TTT 적응 후 예측값
    """
    B      = imgs.shape[0]
    device = imgs.device

    # backbone 제외 trainable params
    ttt_params = [
        p for name, p in model.named_parameters()
        if "base.backbone" not in name and p.requires_grad
    ]

    all_idh, all_codel, all_grade = [], [], []

    for b in range(B):
        img_b = imgs[b:b+1]
        ag_b  = age_g[b:b+1]
        sx_b  = sex_b[b:b+1]

        # 원본 파라미터 저장
        orig_state = copy.deepcopy(model.state_dict())

        if ttt_params:
            opt = torch.optim.Adam(ttt_params, lr=lr)
            model.eval()   # dropout 비활성화 상태에서 적응

            for _ in range(n_steps):
                opt.zero_grad()
                idh_p, codel_p, grade_p = model(img_b, ag_b, sx_b)

                # 엔트로피 최소화: H = -sum(p * log p)
                loss = sum(
                    -(F.softmax(logits, 1) * F.log_softmax(logits, 1)).sum(1).mean()
                    for logits in (idh_p, codel_p, grade_p)
                )
                loss.backward()
                opt.step()

        # 최종 예측
        model.eval()
        with torch.no_grad():
            idh_p, codel_p, grade_p = model(img_b, ag_b, sx_b)
            all_idh.append(F.softmax(idh_p,   dim=1))
            all_codel.append(F.softmax(codel_p, dim=1))
            all_grade.append(F.softmax(grade_p, dim=1))

        # 파라미터 복구
        model.load_state_dict(orig_state)
        model.eval()

    return (
        torch.cat(all_idh,   dim=0),
        torch.cat(all_codel, dim=0),
        torch.cat(all_grade, dim=0),
    )