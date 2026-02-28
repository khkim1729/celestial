"""
HAEDAL_Trainer.py
-----------------
Training loop for HierarchicalGliomaClassifierWithClinical.
Each (subject, modality) sample is treated as an independent sample.
"""

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from HAEDAL_Config import HAEDALConfig
from HAEDAL_ClinicalClassifier import HierarchicalGliomaClassifierWithClinical
from HAEDAL_Loss import HierarchicalGliomaLoss
from HAEDAL_Loader import make_loader
from HAEDAL_Metrics import compute_metrics, print_metrics, history_to_tsv


# ── Early Stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """patience 에포크 동안 개선이 없으면 학습 중단."""
    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = -float("inf")
        self.counter   = 0

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        if value > self.best + self.min_delta:
            self.best    = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── Trainer ────────────────────────────────────────────────────────────────────

class HAEDALTrainer:
    def __init__(self, cfg: HAEDALConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.out_dir  = Path(cfg.output_dir) / cfg.experiment_name
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model = HierarchicalGliomaClassifierWithClinical(
            num_grades=cfg.num_grades,
            freeze_backbone=cfg.freeze_backbone,
            clinical_dim=cfg.clinical_dim,
        ).to(self.device)

        self.criterion = HierarchicalGliomaLoss(
            w_idh=cfg.w_idh, w_codel=cfg.w_codel, w_grade=cfg.w_grade,
        )
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler    = torch.amp.GradScaler("cuda", enabled=cfg.amp)
        self.scheduler = self._make_scheduler()

        self.early_stopper = (
            EarlyStopping(cfg.early_stop_patience, cfg.early_stop_min_delta)
            if cfg.early_stop_patience > 0 else None
        )

        self.best_score = -1e9
        self.history    = {"train": [], "val": []}

    def _make_scheduler(self):
        s = self.cfg.scheduler
        if s == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.cfg.epochs - self.cfg.warmup_epochs),
                eta_min=1e-6,
            )
        if s == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        return None

    def _run_epoch(self, loader, train: bool) -> dict:
        self.model.train(train)
        tot_loss = 0.0
        n = 0
        res = {t: {"y_true": [], "y_pred": [], "y_prob": []} for t in ("idh", "codel", "grade")}

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                imgs    = batch["image"].to(self.device)
                idh_t   = batch["idh"].to(self.device)
                codel_t = batch["codel"].to(self.device)
                grade_t = batch["grade"].to(self.device)
                age_g   = batch["age_group"].to(self.device)
                sex_b   = batch["sex_bin"].to(self.device)

                with torch.amp.autocast("cuda", enabled=self.cfg.amp):
                    idh_p, codel_p, grade_p = self.model(imgs, age_g, sex_b)
                    loss, ld = self.criterion(
                        (idh_p, codel_p, grade_p),
                        (idh_t, codel_t, grade_t),
                    )

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                tot_loss += ld["total"]
                n += 1

                for key, logits, tgt in (
                    ("idh", idh_p, idh_t), ("codel", codel_p, codel_t), ("grade", grade_p, grade_t)
                ):
                    probs = F.softmax(logits.detach(), dim=1).cpu().tolist()
                    preds = logits.detach().argmax(1).cpu().tolist()
                    res[key]["y_true"].extend(tgt.cpu().tolist())
                    res[key]["y_pred"].extend(preds)
                    res[key]["y_prob"].extend(probs)

        metrics = compute_metrics(res)
        metrics["loss"] = tot_loss / max(n, 1)
        return metrics

    def train(self):
        cfg = self.cfg
        print(f"\n{'='*60}\n  HAEDAL | {cfg.experiment_name} | {self.device}\n{'='*60}")
        if self.early_stopper:
            print(f"  EarlyStopping: patience={cfg.early_stop_patience}  "
                  f"min_delta={cfg.early_stop_min_delta}  "
                  f"monitor={cfg.save_best_metric}")
        print()

        train_loader = make_loader(cfg.train_csv, cfg, split="train")
        val_loader   = make_loader(cfg.val_csv,   cfg, split="val")

        train_start = datetime.now()
        epoch_times = []

        for ep in range(1, cfg.epochs + 1):
            # ── 에포크 타이밍 ──
            t_train_start = datetime.now()
            tr = self._run_epoch(train_loader, train=True)
            t_train_stop  = datetime.now()

            t_val_start = datetime.now()
            vl = self._run_epoch(val_loader, train=False)
            t_val_stop  = datetime.now()

            epoch_times.append({
                "train_start": t_train_start, "train_stop": t_train_stop,
                "val_start":   t_val_start,   "val_stop":   t_val_stop,
            })

            if self.scheduler and ep > cfg.warmup_epochs:
                self.scheduler.step()

            score    = vl.get("overall", {}).get("score", 0.0)
            idh_auc  = vl.get("idh",     {}).get("auc",   0.0)
            mean_acc = vl.get("overall", {}).get("mean_acc", 0.0)
            ep_sec   = (t_val_stop - t_train_start).total_seconds()

            print(f"[{ep:03d}/{cfg.epochs}] "
                  f"loss {tr['loss']:.4f}→{vl['loss']:.4f}  "
                  f"idh_auc={idh_auc:.4f}  acc={mean_acc:.4f}  "
                  f"score={score:.4f}  ({ep_sec:.1f}s)")

            self.history["train"].append(tr)
            self.history["val"].append(vl)
            self._save(ep, score, "latest.pt")

            cur = self._pick(vl, cfg.save_best_metric)
            if cur > self.best_score:
                self.best_score = cur
                self._save(ep, score, "best.pt")
                print(f"  ✓ best ({cfg.save_best_metric}={cur:.4f})")

            # ── Early stopping ──
            if self.early_stopper and self.early_stopper.step(cur):
                print(f"  ✗ Early stop (no improvement for "
                      f"{cfg.early_stop_patience} epochs)")
                break

        train_stop = datetime.now()
        total_sec  = (train_stop - train_start).total_seconds()
        print(f"\nDone. best_score={self.best_score:.4f}  "
              f"total={total_sec/60:.1f}min  "
              f"start={train_start.strftime('%H:%M:%S')}  "
              f"stop={train_stop.strftime('%H:%M:%S')}")

        # ── 저장 ──
        with open(self.out_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)

        tsv_path = self.out_dir / "history_metrics.tsv"
        history_to_tsv(
            self.history, tsv_path,
            epoch_times=epoch_times,
            train_start=train_start,
            train_stop=train_stop,
        )
        print(f"[Trainer] TSV → {tsv_path}")

    def _pick(self, metrics: dict, key: str) -> float:
        # "task_metric" 형식 우선 처리 (예: "idh_auc" → metrics["idh"]["auc"])
        for task in ("idh", "codel", "grade", "overall"):
            if key.startswith(task + "_"):
                sub_key = key[len(task) + 1:]
                v = metrics.get(task, {}).get(sub_key)
                if v is not None and isinstance(v, (int, float)):
                    return float(v)
        # 폴백: 모든 서브 dict에서 key 직접 검색
        for v in metrics.values():
            if isinstance(v, dict) and key in v:
                r = v[key]
                return float(r) if isinstance(r, (int, float)) else 0.0
        return 0.0

    def _save(self, epoch: int, score: float, name: str):
        torch.save({
            "epoch": epoch, "score": score,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": vars(self.cfg),
        }, self.ckpt_dir / name)
