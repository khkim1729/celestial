import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalGliomaLoss(nn.Module):
    def __init__(self, w_idh=3.0, w_codel=1.0, w_grade=1.0): #일부로 idh w 세게
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none') # 샘플별 손실 계산을 위해 none 설정 focal loss도 사용할만두...
        self.w_idh = w_idh
        self.w_codel = w_codel
        self.w_grade = w_grade

    def forward(self, preds, targets):
        idh_p, codel_p, grade_p = preds
        idh_t, codel_t, grade_t = targets

        # 1. 유효 샘플 마스크 (label == -1 은 결측치)
        idh_valid   = (idh_t   >= 0).float()  # [B]
        codel_valid = (codel_t >= 0).float()  # [B]
        grade_valid = (grade_t >= 0).float()  # [B]

        # 2. CrossEntropy — -1을 0으로 clamp 후 마스크로 해당 손실 제거
        #    (CE 자체는 clamp된 임시값으로 계산, invalid 기여분은 0으로 소거)
        l_idh   = self.ce(idh_p,   idh_t.clamp(min=0))   * idh_valid    # [B]
        l_codel = self.ce(codel_p, codel_t.clamp(min=0)) * codel_valid  # [B]
        l_grade = self.ce(grade_p, grade_t.clamp(min=0)) * grade_valid  # [B]

        # 3. IDH dependency mask
        #    IDH label이 없는 샘플은 wrong=0 처리 (dependency_mask=1 유지)
        with torch.no_grad():
            idh_pred_labels   = torch.argmax(idh_p,   dim=1)
            codel_pred_labels = torch.argmax(codel_p, dim=1)

            is_idh_wrong = torch.zeros(idh_t.shape[0], device=idh_t.device)
            known = idh_valid.bool()
            if known.any():
                is_idh_wrong[known] = (
                    idh_pred_labels[known] != idh_t[known]
                ).float()
            dependency_mask = 1.0 - (is_idh_wrong * 0.8)

            # IDH-wt 예측 & 1p19q-codel 예측 → 불일치 페널티 (예측만으로 결정)
            inconsistent  = ((idh_pred_labels == 0) & (codel_pred_labels == 1)).float()
            penalty_const = inconsistent * 2.0

        # 4. 유효 샘플 수 기준 평균 (최소 1로 clamp)
        n_idh   = idh_valid.sum().clamp(min=1)
        n_codel = codel_valid.sum().clamp(min=1)
        n_grade = grade_valid.sum().clamp(min=1)

        loss_idh    = l_idh.sum()                       / n_idh   * self.w_idh
        loss_codel  = (l_codel * dependency_mask).sum() / n_codel * self.w_codel
        loss_grade  = (l_grade * dependency_mask).sum() / n_grade * self.w_grade
        loss_penalty = penalty_const.mean()

        total_loss = loss_idh + loss_codel + loss_grade + loss_penalty

        return total_loss, {
            'total':   total_loss.item(),
            'idh':     loss_idh.item(),
            'codel':   loss_codel.item(),
            'grade':   loss_grade.item(),
            'penalty': loss_penalty.item(),
        }