import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalGliomaClassifier(nn.Module):
    def __init__(self, num_grades=3, freeze_backbone=True):
        super().__init__()
        # 1. DINOv2 로드 (A100 최적화: Giant 모델)
        # xFormers 미설치 시 표준 PyTorch attention으로 자동 폴백 (동작에 문제 없음)
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        embed_dim = 1536  # ViT-g의 출력 차원(github에 명시되어 있음)
        
        # 2. Fine-tuning On/Off 설정 (일단 시간상 freeze가 맞는데 finetuning하면 정확도는 높아질거라)
        self.set_backbone_freeze(freeze_backbone)

        # 3. Task 1: IDH Status (상위 단계)
        self.idh_head = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 2) # Mutant vs Wild-type
        )

        # 4. Task 2: 1p/19q Co-deletion (IDH 결과에 의존)
        self.idh_embed = nn.Linear(2, 128) # IDH 결과를 특징으로 변환
        self.codel_head = nn.Sequential(
            nn.Linear(embed_dim + 128, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

        # 5. Task 3: WHO Grade (IDH & 1p19q 결과에 모두 의존)
        self.genetic_embed = nn.Linear(4, 256) # IDH(2) + 1p19q(2) 결과를 통합
        self.grade_head = nn.Sequential(
            nn.Linear(embed_dim + 256, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_grades)
        )

    def set_backbone_freeze(self, freeze=True):
        """Fine-tuning 여부를 결정하는 스위치"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        print(f"--- Backbone Freeze: {freeze} (Training: {not freeze}) ---")

    def forward(self, x):
        # 3채널 입력 필요 (DINOv2 규격이라) 
        shared_features = self.backbone(x)

        # Step 1: IDH
        idh_logits = self.idh_head(shared_features)
        
        # Step 2: 1p19q (IDH 정보를 컨텍스트로 주입)
        idh_info = F.gelu(self.idh_embed(idh_logits))
        codel_input = torch.cat([shared_features, idh_info], dim=-1)
        codel_logits = self.codel_head(codel_input)

        # Step 3: Grade (IDH + 1p19q 통합 정보를 컨텍스트로 주입)
        genetic_info = F.gelu(self.genetic_embed(torch.cat([idh_logits, codel_logits], dim=-1)))
        grade_input = torch.cat([shared_features, genetic_info], dim=-1)
        grade_logits = self.grade_head(grade_input)

        return idh_logits, codel_logits, grade_logits
