"""
HAEDAL_ClinicalClassifier.py
-----------------------------
HierarchicalGliomaClassifier + ClinicalEncoder.

Clinical features:
  age_group : 0 = Young (age < AGE_CUTOFF)   1 = Old (age >= AGE_CUTOFF)
  sex       : 0 = Female                      1 = Male

Architecture:
  img_feat  [B, 1536]  ← DINOv2 backbone (from base classifier)
  clin_feat [B, D]     ← ClinicalEncoder(age_group, sex)
  fused     [B, 1536+D]

  fused → idh_head → idh_logits
       ↓ + idh_context
       → codel_head → codel_logits
       ↓ + genetic_context (idh + codel)
       → grade_head → grade_logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from HAEDAL_Classifier import HierarchicalGliomaClassifier

AGE_CUTOFF = 45


class ClinicalEncoder(nn.Module):
    """Binary clinical features → dense embedding."""
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )

    def forward(self, age_group: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        """
        age_group : [B] long  (0=Young / 1=Old)
        sex       : [B] long  (0=F    / 1=M)
        returns   : [B, output_dim]
        """
        x = torch.stack([age_group.float(), sex.float()], dim=-1)  # [B, 2]
        return self.encoder(x)


class HierarchicalGliomaClassifierWithClinical(nn.Module):
    """
    Drop-in replacement for HierarchicalGliomaClassifier.
    forward(x, age_group, sex) → (idh_logits, codel_logits, grade_logits)
    """
    def __init__(self, num_grades: int = 3, freeze_backbone: bool = True,
                 clinical_dim: int = 64):
        super().__init__()
        self.base = HierarchicalGliomaClassifier(
            num_grades=num_grades, freeze_backbone=freeze_backbone
        )
        self.clinical = ClinicalEncoder(output_dim=clinical_dim)

        embed_dim = 1536
        fused_dim = embed_dim + clinical_dim

        # Hierarchical heads (mirror base classifier, input width = fused_dim)
        self.idh_head = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 2),
        )
        self.idh_embed     = nn.Linear(2, 128)
        self.codel_head    = nn.Sequential(
            nn.Linear(fused_dim + 128, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 2),
        )
        self.genetic_embed = nn.Linear(4, 256)
        self.grade_head    = nn.Sequential(
            nn.Linear(fused_dim + 256, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_grades),
        )

    def set_backbone_freeze(self, freeze: bool = True):
        self.base.set_backbone_freeze(freeze)

    def forward(self, x: torch.Tensor,
                age_group: torch.Tensor,
                sex: torch.Tensor):
        img_feat  = self.base.backbone(x)                        # [B, 1536]
        clin_feat = self.clinical(age_group, sex)                # [B, clinical_dim]
        fused     = torch.cat([img_feat, clin_feat], dim=-1)    # [B, fused_dim]

        idh_logits   = self.idh_head(fused)
        idh_info     = F.relu(self.idh_embed(idh_logits))
        codel_logits = self.codel_head(torch.cat([fused, idh_info], dim=-1))
        genetic_info = F.relu(self.genetic_embed(
            torch.cat([idh_logits, codel_logits], dim=-1)
        ))
        grade_logits = self.grade_head(torch.cat([fused, genetic_info], dim=-1))

        return idh_logits, codel_logits, grade_logits


def encode_clinical(age: float, sex_str: str,
                    age_cutoff: int = AGE_CUTOFF) -> tuple[int, int]:
    """
    Convert raw CSV values to binary integers.
      age      : float  → 0 (Young) / 1 (Old)
      sex_str  : "M"/"F" (case-insensitive) → 0 (F) / 1 (M)
    """
    age_group = 1 if float(age) >= age_cutoff else 0
    sex       = 1 if str(sex_str).strip().upper() == "M" else 0
    return age_group, sex
