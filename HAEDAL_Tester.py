"""
HAEDAL_Tester.py
----------------
Test evaluation for HierarchicalGliomaClassifier.
Each (patient, axis) sample is evaluated independently — same as training.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from HAEDAL_Config import HAEDALConfig
from HAEDAL_ClinicalClassifier import HierarchicalGliomaClassifierWithClinical
from HAEDAL_Loss import HierarchicalGliomaLoss
from HAEDAL_Loader import make_loader
from HAEDAL_Metrics import compute_metrics, print_metrics, metrics_to_tsv


class HAEDALTester:
    def __init__(self, cfg: HAEDALConfig, checkpoint: str):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(cfg.output_dir) / cfg.experiment_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.model = HierarchicalGliomaClassifierWithClinical(
            num_grades=cfg.num_grades,
            freeze_backbone=cfg.freeze_backbone,
            clinical_dim=cfg.clinical_dim,
        ).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        print(f"[Tester] {checkpoint}  epoch={ckpt.get('epoch','?')} score={ckpt.get('score',0):.4f}")

        self.criterion = HierarchicalGliomaLoss(
            w_idh=cfg.w_idh, w_codel=cfg.w_codel, w_grade=cfg.w_grade,
        )

    @torch.no_grad()
    def evaluate(self, test_csv: str) -> dict:
        loader = make_loader(test_csv, self.cfg, split="test")
        res = {t: {"y_true": [], "y_pred": [], "y_prob": []} for t in ("idh", "codel", "grade")}
        rows = []
        tot_loss, nb = 0.0, 0

        start_time = datetime.now()
        for batch in loader:
            imgs    = batch["image"].to(self.device)
            idh_t   = batch["idh"].to(self.device)
            codel_t = batch["codel"].to(self.device)
            grade_t = batch["grade"].to(self.device)
            age_g   = batch["age_group"].to(self.device)
            sex_b   = batch["sex_bin"].to(self.device)

            idh_p, codel_p, grade_p = self.model(imgs, age_g, sex_b)
            loss, _ = self.criterion(
                (idh_p, codel_p, grade_p), (idh_t, codel_t, grade_t)
            )
            tot_loss += loss.item(); nb += 1

            for key, logits, tgt in (
                ("idh", idh_p, idh_t), ("codel", codel_p, codel_t), ("grade", grade_p, grade_t)
            ):
                probs = F.softmax(logits, dim=1).cpu().tolist()
                preds = logits.argmax(1).cpu().tolist()
                res[key]["y_true"].extend(tgt.cpu().tolist())
                res[key]["y_pred"].extend(preds)
                res[key]["y_prob"].extend(probs)

            for i, sid in enumerate(batch["subject_id"]):
                rows.append({
                    "subject_id":  sid,
                    "modality":    batch["modality"][i],
                    "idh_label":   idh_t[i].item(),
                    "idh_pred":    idh_p[i].argmax().item(),
                    "idh_prob_mut": round(float(F.softmax(idh_p[i], dim=0)[1]), 4),
                    "codel_label": codel_t[i].item(),
                    "codel_pred":  codel_p[i].argmax().item(),
                    "codel_prob":  round(float(F.softmax(codel_p[i], dim=0)[1]), 4),
                    "grade_label": grade_t[i].item(),
                    "grade_pred":  grade_p[i].argmax().item(),
                })

        stop_time = datetime.now()
        metrics = compute_metrics(res)
        metrics["test_loss"] = tot_loss / max(nb, 1)

        print(f"\n{'='*60}\n  TEST — {self.cfg.experiment_name}  loss={metrics['test_loss']:.4f}\n{'='*60}")
        print_metrics(metrics)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if rows:
            p = self.out_dir / f"test_results_{ts}.csv"
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
            print(f"[Tester] CSV  → {p}")

        p2 = self.out_dir / f"test_metrics_{ts}.json"
        with open(p2, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"[Tester] JSON → {p2}")

        p3 = self.out_dir / f"test_metrics_{ts}.tsv"
        metrics_to_tsv(metrics, p3,
                       start_time=start_time, stop_time=stop_time,
                       n_samples=len(loader.dataset))
        print(f"[Tester] TSV  → {p3}")
        return metrics

    def gradcam(self, test_csv: str, max_samples: int | None = None) -> None:
        """
        Grad-CAM 시각화 생성. (requires: pip install grad-cam)

        각 (subject, modality) 샘플에 대해 3 tasks × 3 axes 히트맵을
        output/{exp}/gradcam/{subject}_{modality}.png 로 저장.

        max_samples : 생성할 최대 샘플 수 (None = 전체)
        """
        from HAEDAL_GradCAM import generate_gradcam

        cam_dir = self.out_dir / "gradcam"
        cam_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[GradCAM] 저장 경로: {cam_dir}")
        if max_samples:
            print(f"[GradCAM] 최대 {max_samples}개 샘플")

        loader = make_loader(test_csv, self.cfg, split="test")
        self.model.eval()

        count = 0
        for batch in loader:
            for i in range(len(batch["subject_id"])):
                if max_samples is not None and count >= max_samples:
                    print(f"[GradCAM] 완료 ({count}개)")
                    return

                img = batch["image"][i:i+1].to(self.device)
                ag  = batch["age_group"][i:i+1].to(self.device)
                sx  = batch["sex_bin"][i:i+1].to(self.device)

                # 예측값 취득 (no_grad로 빠르게)
                with torch.no_grad():
                    idh_p, codel_p, grade_p = self.model(img, ag, sx)

                pred = {
                    "idh":   int(idh_p.argmax(1).item()),
                    "codel": int(codel_p.argmax(1).item()),
                    "grade": int(grade_p.argmax(1).item()),
                }
                true = {
                    "idh":   int(batch["idh"][i].item()),
                    "codel": int(batch["codel"][i].item()),
                    "grade": int(batch["grade"][i].item()),
                }

                sid = batch["subject_id"][i]
                mod = batch["modality"][i]
                fname = f"{sid}_{mod}.png"

                generate_gradcam(
                    model        = self.model,
                    img_tensor   = img,
                    age_group    = ag,
                    sex          = sx,
                    pred_classes = pred,
                    true_classes = true,
                    save_path    = cam_dir / fname,
                    title        = f"{sid}  {mod}",
                )
                count += 1
                print(f"  [{count:>4}] {fname}"
                      f"  IDH:{true['idh']}→{pred['idh']}"
                      f"  1p19q:{true['codel']}→{pred['codel']}"
                      f"  Grade:{true['grade']}→{pred['grade']}")

        print(f"[GradCAM] 완료 ({count}개)  → {cam_dir}")
