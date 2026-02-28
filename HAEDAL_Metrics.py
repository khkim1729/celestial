"""
HAEDAL_Metrics.py
-----------------
Comprehensive per-task metrics for the HAEDAL multi-task classifier.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, classification_report,
)

TASK_CLASSES = {
    "idh":   ["IDH-wt",    "IDH-mut"],
    "codel": ["Non-codel", "Codeleted"],
    "grade": ["Grade2",    "Grade3",   "Grade4"],
}
TASK_LABEL = {"idh": "IDH", "codel": "1p/19q", "grade": "Grade"}


def task_metrics(y_true, y_pred, y_prob, task: str) -> Dict:
    valid = np.array(y_true) >= 0
    if not valid.any():
        return {}
    yt = np.array(y_true)[valid]
    yp = np.array(y_pred)[valid]
    ypr = np.array(y_prob)[valid]
    classes = TASK_CLASSES[task]
    binary = len(classes) == 2
    avg = "binary" if binary else "macro"

    m = {
        "n":           int(valid.sum()),
        "acc":         float(accuracy_score(yt, yp)),
        "bal_acc":     float(balanced_accuracy_score(yt, yp)),
        "f1_macro":    float(f1_score(yt, yp, average="macro",    zero_division=0)),
        "f1_weighted": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "precision":   float(precision_score(yt, yp, average=avg, zero_division=0)),
        "recall":      float(recall_score(yt, yp,    average=avg, zero_division=0)),
        "mcc":         float(matthews_corrcoef(yt, yp)),
        "kappa":       float(cohen_kappa_score(yt, yp)),
    }
    # Per-class F1
    for cls, f1 in zip(classes, f1_score(yt, yp, average=None, zero_division=0)):
        m[f"f1_{cls.replace(' ','_').lower()}"] = float(f1)

    # AUC
    try:
        if binary:
            m["auc"] = float(roc_auc_score(yt, ypr[:, 1]))
        elif len(set(yt)) > 1:
            m["auc_ovr"] = float(roc_auc_score(yt, ypr, multi_class="ovr", average="macro"))
            m["auc_ovo"] = float(roc_auc_score(yt, ypr, multi_class="ovo", average="macro"))
    except Exception:
        pass

    m["confusion_matrix"] = confusion_matrix(yt, yp).tolist()
    m["report"] = classification_report(yt, yp, target_names=classes, zero_division=0)
    return m


def compute_metrics(results: Dict) -> Dict:
    """
    results = {
        "idh":   {"y_true": [...], "y_pred": [...], "y_prob": [[...], ...]},
        "codel": {...}, "grade": {...}
    }
    """
    out = {}
    for task in ("idh", "codel", "grade"):
        if task in results:
            r = results[task]
            out[task] = task_metrics(r["y_true"], r["y_pred"], r["y_prob"], task)

    accs = [m["acc"] for m in out.values() if "acc" in m]
    idh_auc = out.get("idh", {}).get("auc")
    mean_acc = float(np.mean(accs)) if accs else 0.0
    out["overall"] = {
        "mean_acc": round(mean_acc, 4),
        "score":    round((mean_acc + idh_auc) / 2, 4) if idh_auc else mean_acc,
    }
    return out


def _fmt(dt) -> str:
    """datetime → 'YYYY-MM-DD HH:MM:SS' 문자열."""
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt is not None else ""


def metrics_to_tsv(metrics: Dict, path,
                   start_time=None, stop_time=None,
                   n_samples: int = None) -> None:
    """Flatten metrics dict → TSV (task / metric / value).
    timing 행: task=_timing  (start_time / stop_time / inference_sec / n_samples)
    Skips 'report'. Serialises confusion_matrix as a string."""
    with open(path, "w", newline="") as f:
        f.write("task\tmetric\tvalue\n")
        # timing 행
        if start_time is not None:
            f.write(f"_timing\tstart_time\t{_fmt(start_time)}\n")
        if stop_time is not None:
            f.write(f"_timing\tstop_time\t{_fmt(stop_time)}\n")
        if start_time is not None and stop_time is not None:
            sec = (stop_time - start_time).total_seconds()
            f.write(f"_timing\tinference_sec\t{sec:.3f}\n")
        if n_samples is not None:
            f.write(f"_timing\tn_samples\t{n_samples}\n")
        # 지표 행
        for task, m in metrics.items():
            if not isinstance(m, dict):
                continue
            for k, v in m.items():
                if k == "report":
                    continue
                if k == "confusion_matrix":
                    v = str(v)
                f.write(f"{task}\t{k}\t{v}\n")


def history_to_tsv(history: Dict, path,
                   epoch_times: list = None,
                   train_start=None, train_stop=None) -> None:
    """Flatten training history → TSV (epoch / split / task / metric / value).
    epoch=0 / split=overall / task=_timing : 전체 학습 타이밍
    epoch=N / split=train|val / task=_timing : 에포크별 타이밍
    Skips 'report' and 'confusion_matrix'.

    epoch_times: [{"train_start": dt, "train_stop": dt,
                   "val_start": dt,   "val_stop": dt}, ...]  (1-indexed)
    """
    with open(path, "w", newline="") as f:
        f.write("epoch\tsplit\ttask\tmetric\tvalue\n")
        # 전체 학습 타이밍 (epoch=0)
        if train_start is not None:
            f.write(f"0\toverall\t_timing\ttrain_start\t{_fmt(train_start)}\n")
        if train_stop is not None:
            f.write(f"0\toverall\t_timing\ttrain_stop\t{_fmt(train_stop)}\n")
        if train_start is not None and train_stop is not None:
            total = (train_stop - train_start).total_seconds()
            f.write(f"0\toverall\t_timing\ttotal_sec\t{total:.3f}\n")
        # 에포크별
        for split in ("train", "val"):
            for ep_idx, metrics in enumerate(history.get(split, []), start=1):
                # 타이밍
                if epoch_times and ep_idx <= len(epoch_times):
                    et = epoch_times[ep_idx - 1]
                    k_start, k_stop = f"{split}_start", f"{split}_stop"
                    if k_start in et:
                        f.write(f"{ep_idx}\t{split}\t_timing\tstart_time\t{_fmt(et[k_start])}\n")
                    if k_stop in et:
                        f.write(f"{ep_idx}\t{split}\t_timing\tstop_time\t{_fmt(et[k_stop])}\n")
                    if k_start in et and k_stop in et:
                        dur = (et[k_stop] - et[k_start]).total_seconds()
                        f.write(f"{ep_idx}\t{split}\t_timing\tduration_sec\t{dur:.3f}\n")
                # 지표
                for task, m in metrics.items():
                    if not isinstance(m, dict):
                        continue
                    for k, v in m.items():
                        if k in ("report", "confusion_matrix"):
                            continue
                        f.write(f"{ep_idx}\t{split}\t{task}\t{k}\t{v}\n")


def print_metrics(all_metrics: Dict):
    for task, m in all_metrics.items():
        if not isinstance(m, dict):
            continue
        print(f"\n── {TASK_LABEL.get(task, task.upper())} " + "─" * 40)
        for k, v in m.items():
            if k == "confusion_matrix":
                print(f"  confusion_matrix:")
                for row in v:
                    print(f"    {row}")
            elif k == "report":
                print(f"  classification_report:\n{v}")
            elif isinstance(v, float):
                print(f"  {k:<20}: {v:.4f}")
            else:
                print(f"  {k:<20}: {v}")
