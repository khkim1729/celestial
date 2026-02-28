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

        # 1. 기본 CrossEntropy
        l_idh = self.ce(idh_p, idh_t)
        l_codel = self.ce(codel_p, codel_t)
        l_grade = self.ce(grade_p, grade_t)

        # 2. IDH 페널티
        # IDH 예측이 틀린 샘플 식별 (Pred != Target)
        with torch.no_grad():
            idh_pred_labels = torch.argmax(idh_p, dim=1)
            is_idh_wrong = (idh_pred_labels != idh_t).float() # 틀리면 1, 맞으면 0

        # IDH가 틀린 샘플은 하위 태스크 손실을 0.2배로 억제 (학습 방해 방지)
        # 반대로 말하면 IDH를 맞춰야만 하위 태스크가 제대로 학습됨
        dependency_mask = 1.0 - (is_idh_wrong * 0.8) 

        # 3. IDH - 1p19q 페널티
        # IDH-wildtype(0)인데 1p19q-codel(1)로 예측하는 경우 페널티 부여
        # WHO 가이드라인: IDH-wt은 항상 1p19q non-codel임
        with torch.no_grad():
            codel_pred_labels = torch.argmax(codel_p, dim=1)
            inconsistent = ((idh_pred_labels == 0) & (codel_pred_labels == 1)).float()
        
        penalty_const = inconsistent * 2.0 # 불일치 시 강력한 페널티

        # 4. 최종 가중 합산
        loss_idh = l_idh.mean() * self.w_idh
        loss_codel = (l_codel * dependency_mask).mean() * self.w_codel
        loss_grade = (l_grade * dependency_mask).mean() * self.w_grade
        loss_penalty = penalty_const.mean()

        total_loss = loss_idh + loss_codel + loss_grade + loss_penalty

        return total_loss, {
            'total': total_loss.item(),
            'idh': loss_idh.item(),
            'codel': loss_codel.item(),
            'grade': loss_grade.item(),
            'penalty': loss_penalty.item()
        }