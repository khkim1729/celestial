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

        # 3. IDH dependency mask & Penalty (최소 수정 버전)
        
        # 확률값 계산 (softmax)
        idh_probs = torch.softmax(idh_p, dim=1)
        codel_probs = torch.softmax(codel_p, dim=1)

        # 1) Dependency Mask
        target_idx = idh_t.clamp(min=0).unsqueeze(1)
        prob_wrong = 1.0 - idh_probs.gather(1, target_idx).squeeze(1)
        
        # 0.8(ablation 대상)이 이제 학습에 반영됩니다.
        dependency_mask = 1.0 - (prob_wrong * 0.8) 

        # 2) Penalty
        # IDH-wt(0번) 확률 * Codel-1(1번) 확률
        inconsistent_soft = idh_probs[:, 0] * codel_probs[:, 1]
        
        penalty_const = inconsistent_soft * 0.0 

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