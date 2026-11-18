"""Learned multimodal fusion classifier for Hateful Memes (binary).

Pipeline::

    extract features (model.hm_features)   -> data/hateful_memes/.cache/*.npz
    split official TRAIN into train / val   (stratified, grouped by caption/image)
    fit scaler on train rows only
    fit candidate classifiers on train, select by VALIDATION ROC-AUC
    choose the decision threshold on VALIDATION probabilities (frozen afterwards)
    save bundle (scaler + classifier + threshold + column metadata)

The official ``dev`` split is never touched here.  It is evaluated exactly once,
by ``evaluate/hm_eval.py``, using the frozen bundle.

Baselines (one bundle each, same protocol)::

    text    A  RoBERTa embedding + toxic-bert scores
    image   B  CLIP embedding + CLIP policy-concept probabilities
    concat  C  RoBERTa + CLIP embeddings
    policy  D  C + toxic-bert + concept vector + policy aggregates
    fusion  -  D + CLIP caption embedding + image*caption interaction block

(Baseline E, the existing GAT, is trained by ``model/train.py``.)

CLI::

    python -m model.hm_fusion extract --data-dir data/hateful_memes --splits train dev
    python -m model.hm_fusion train   --data-dir data/hateful_memes
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from model.hm_features import FEATURE_SETS, build_matrix, default_cache_path, load_features

SEED = 42
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "hm_fusion")


def _log(message: str) -> None:
    print(f"[CHEM][hm_fusion] {message}")


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Splitting (no leakage between train and val)
# --------------------------------------------------------------------------- #
def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def group_ids(texts: Sequence[str], md5s: Sequence[str]) -> np.ndarray:
    """Group memes that share a caption OR an image (union-find).

    Hateful Memes contains deliberate "confounders" that reuse the same caption
    with a different image (and vice versa).  A random split would put siblings
    on both sides and let the classifier memorise captions, so siblings are
    kept together.
    """
    n = len(texts)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seen: Dict[Tuple[str, str], int] = {}
    for i in range(n):
        for key in (("t", _norm_text(texts[i])), ("i", str(md5s[i]))):
            if key[0] == "t" and not key[1]:
                continue  # empty captions are not a grouping key
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    roots: Dict[int, int] = {}
    return np.asarray([roots.setdefault(find(i), len(roots)) for i in range(n)])


def split_train_val(
    labels: Sequence[int],
    texts: Sequence[str],
    md5s: Sequence[str],
    val_frac: float = 0.15,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified, group-aware train/val split of the OFFICIAL TRAIN split.

    Returns index arrays ``(train_idx, val_idx)``; groups never straddle them.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(labels)
    groups = group_ids(texts, md5s)
    n_splits = max(2, int(round(1.0 / val_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    return np.sort(train_idx), np.sort(val_idx)


# --------------------------------------------------------------------------- #
# Metrics / threshold / calibration
# --------------------------------------------------------------------------- #
def metric_report(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> Dict[str, Any]:
    """Full binary report at ``threshold`` (positive class = HARMFUL = 1)."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2

    def _div(a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0

    return {
        "n_samples": int(len(y)),
        "n_safe": int((y == 0).sum()),
        "n_harmful": int((y == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "safe_precision": _div(tn, tn + fn),
        "safe_recall": _div(tn, tn + fp),
        "harmful_precision": _div(tp, tp + fp),
        "harmful_recall": _div(tp, tp + fn),
        "predicted_harmful_rate": float(pred.mean()),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def best_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    objective: str = "balanced_accuracy",
) -> float:
    """Grid-search 0.05..0.95 (step 0.01) on VALIDATION probabilities.

    ``balanced_accuracy`` (default) averages SAFE and HARMFUL recall, so it
    cannot be gamed by predicting one class and does not depend on the class
    prior of the split it is tuned on (the official train split is ~36 % hateful
    while dev is ~50 %).  ``f1`` and ``accuracy`` are also available.
    Ties resolve to the threshold closest to 0.5.
    """
    if objective not in ("balanced_accuracy", "f1", "accuracy"):
        raise ValueError("objective must be balanced_accuracy, f1 or accuracy")
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    best_t, best_v = 0.5, -1.0
    for i in range(5, 96):
        t = i / 100.0
        r = metric_report(y, p, t)
        v = 0.5 * (r["safe_recall"] + r["harmful_recall"]) if objective == "balanced_accuracy" else r[objective]
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_v = t, v
    return best_t


def calibration_table(y_true: Sequence[int], y_prob: Sequence[float], bins: int = 10) -> Dict[str, Any]:
    """Reliability table and expected calibration error (ECE)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_prob": conf, "frac_harmful": acc})
    return {"ece": float(ece), "bins": rows}


# --------------------------------------------------------------------------- #
# Candidate classifiers
# --------------------------------------------------------------------------- #
def _candidates(seed: int) -> List[Tuple[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    cands: List[Tuple[str, Any]] = []
    for c in (0.003, 0.01, 0.03, 0.1):
        cands.append((f"logreg_C{c}", LogisticRegression(
            C=c, class_weight="balanced", max_iter=3000, random_state=seed)))
    for a in (1e-2, 1e-1, 1.0):
        cands.append((f"mlp_alpha{a}", MLPClassifier(
            hidden_layer_sizes=(256,), alpha=a, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=8, max_iter=200, random_state=seed)))
    return cands


def _fit(name: str, est: Any, X: np.ndarray, y: np.ndarray, seed: int) -> Any:
    """Fit with class balancing (weights for MLP; ``class_weight`` already set for LR)."""
    if name.startswith("mlp"):
        counts = np.bincount(y, minlength=2).astype(float)
        w = (len(y) / (2.0 * counts))[y]
        try:
            est.fit(X, y, sample_weight=w)
            return est
        except TypeError:  # older scikit-learn: no sample_weight for MLP -> oversample minority
            rng = np.random.RandomState(seed)
            minority = int(np.argmin(counts))
            idx_min = np.flatnonzero(y == minority)
            extra = rng.choice(idx_min, size=int(abs(counts[0] - counts[1])), replace=True)
            order = rng.permutation(len(y) + len(extra))
            X2, y2 = np.concatenate([X, X[extra]]), np.concatenate([y, y[extra]])
            est.fit(X2[order], y2[order])
            return est
    est.fit(X, y)
    return est


# --------------------------------------------------------------------------- #
# Bundle (what the evaluator and the API load)
# --------------------------------------------------------------------------- #
@dataclass
class FusionBundle:
    """Frozen inference bundle: scaler + classifier + threshold + column metadata."""

    feature_set: str
    scaler: Any
    clf: Any
    threshold: float
    meta: Dict[str, Any]
    info: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """Harmful-class probability for a batch of feature blocks."""
        X = self.scaler.transform(build_matrix(blocks, self.feature_set))
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        import joblib

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FusionBundle":
        """Load a bundle. Uses pickle: only load files you created yourself."""
        import sys

        import joblib

        # Bundles written by `python -m model.hm_fusion train` record their class
        # as `__main__.FusionBundle`. Under uvicorn/pytest `__main__` is another
        # module, so make that name resolve to this class for the duration of the load.
        main = sys.modules.get("__main__")
        had = hasattr(main, "FusionBundle")
        if main is not None and not had:
            main.FusionBundle = FusionBundle
        try:
            return joblib.load(path)
        finally:
            if main is not None and not had:
                delattr(main, "FusionBundle")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_feature_set(
    blocks: Dict[str, np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_set: str,
    meta: Dict[str, Any],
    seed: int = SEED,
    objective: str = "balanced_accuracy",
) -> Tuple[FusionBundle, Dict[str, Any]]:
    """Train all candidates for one feature set; select on validation ROC-AUC."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X = build_matrix(blocks, feature_set)
    Xtr, Xva = X[train_idx], X[val_idx]
    ytr, yva = labels[train_idx], labels[val_idx]

    scaler = StandardScaler().fit(Xtr)  # train rows only: no val/dev statistics
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

    best: Optional[Tuple[float, str, Any, np.ndarray]] = None
    tried: Dict[str, float] = {}
    for name, est in _candidates(seed):
        set_seed(seed)
        est = _fit(name, est, Xtr_s, ytr, seed)
        pva = est.predict_proba(Xva_s)[:, 1]
        auc = float(roc_auc_score(yva, pva))
        tried[name] = auc
        if best is None or auc > best[0]:
            best = (auc, name, est, pva)
    assert best is not None
    auc, name, est, pva = best

    thr = best_threshold(yva, pva, objective)
    report = metric_report(yva, pva, thr)
    report["chosen_model"] = name
    report["candidates_val_auc"] = tried
    report["calibration_val"] = calibration_table(yva, pva)
    report["n_features"] = int(X.shape[1])
    report["n_train"] = int(len(train_idx))
    report["n_val"] = int(len(val_idx))

    bundle = FusionBundle(
        feature_set=feature_set, scaler=scaler, clf=est, threshold=float(thr), meta=meta,
        info={"seed": seed, "objective": objective, "chosen_model": name,
              "val": {k: report[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}},
    )
    return bundle, report


def train_feature_set_cv(
    blocks: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    feature_set: str,
    meta: Dict[str, Any],
    seed: int = SEED,
    objective: str = "balanced_accuracy",
    n_folds: int = 5,
) -> Tuple[FusionBundle, Dict[str, Any]]:
    """Select model + threshold from OUT-OF-FOLD predictions over all TRAIN rows.

    A single ~15 % validation split of a few thousand memes is noisy; k-fold
    (caption/image-grouped, stratified) gives every train row an out-of-fold
    probability, so model choice and the threshold rest on the whole train set.
    The scaler is refit inside each fold; the final model is refit on all train
    rows.  Dev is never touched.
    """
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    X = build_matrix(blocks, feature_set)
    folds = list(StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, labels, groups))

    best: Optional[Tuple[float, str, Any, np.ndarray]] = None
    tried: Dict[str, float] = {}
    for name, est0 in _candidates(seed):
        oof = np.zeros(len(labels))
        for tr, te in folds:
            set_seed(seed)
            sc = StandardScaler().fit(X[tr])
            est = _fit(name, clone(est0), sc.transform(X[tr]), labels[tr], seed)
            oof[te] = est.predict_proba(sc.transform(X[te]))[:, 1]
        auc = float(roc_auc_score(labels, oof))
        tried[name] = auc
        if best is None or auc > best[0]:
            best = (auc, name, est0, oof)
    assert best is not None
    _, name, est0, oof = best

    thr = best_threshold(labels, oof, objective)
    report = metric_report(labels, oof, thr)
    report.update(chosen_model=name, candidates_val_auc=tried, calibration_val=calibration_table(labels, oof),
                  n_features=int(X.shape[1]), n_train=int(len(labels)), n_val=int(len(labels)), cv_folds=n_folds)

    set_seed(seed)
    scaler = StandardScaler().fit(X)
    final = _fit(name, clone(est0), scaler.transform(X), labels, seed)
    bundle = FusionBundle(
        feature_set=feature_set, scaler=scaler, clf=final, threshold=float(thr), meta=meta,
        info={"seed": seed, "objective": objective, "chosen_model": name, "cv_folds": n_folds,
              "val": {k: report[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}},
    )
    return bundle, report


def train_all(
    data_dir: str,
    out_dir: str = DEFAULT_OUT_DIR,
    seed: int = SEED,
    objective: str = "balanced_accuracy",
    limit: Optional[int] = None,
    feature_sets: Optional[Sequence[str]] = None,
    cv_folds: int = 0,
) -> Dict[str, Any]:
    """Train every baseline + the fusion model from the cached TRAIN features.

    ``cv_folds >= 2`` selects model + threshold from out-of-fold predictions on
    all train rows (see :func:`train_feature_set_cv`); otherwise a single
    train/val split is used.
    """
    set_seed(seed)
    path = default_cache_path(data_dir, "train", limit)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found. Run: python -m model.hm_fusion extract --splits train")
    feats = load_features(path)
    labels = feats["labels"].astype(int)
    if (labels < 0).any():
        raise ValueError("Train split contains unlabeled rows (label -1).")

    if cv_folds >= 2:
        groups = group_ids(feats["texts"], feats["img_md5"])
        _log(f"{cv_folds}-fold grouped CV on train={len(labels)} (harmful {labels.mean():.1%}) | dev untouched")
    else:
        train_idx, val_idx = split_train_val(labels, feats["texts"], feats["img_md5"], seed=seed)
        assert not set(train_idx) & set(val_idx)
        _log(f"train={len(train_idx)} (harmful {labels[train_idx].mean():.1%}) | "
             f"val={len(val_idx)} (harmful {labels[val_idx].mean():.1%}) | dev untouched")

    os.makedirs(out_dir, exist_ok=True)
    summary: Dict[str, Any] = {"seed": seed, "objective": objective, "cv_folds": cv_folds, "sets": {}}
    bundles: Dict[str, FusionBundle] = {}
    for fs in (feature_sets or FEATURE_SETS):
        _log(f"--- feature set '{fs}' ---")
        if cv_folds >= 2:
            bundle, rep = train_feature_set_cv(feats, labels, groups, fs, feats["meta"], seed, objective, cv_folds)
        else:
            bundle, rep = train_feature_set(feats, labels, train_idx, val_idx, fs, feats["meta"], seed, objective)
        bundle.save(os.path.join(out_dir, f"{fs}.joblib"))
        bundles[fs] = bundle
        summary["sets"][fs] = rep
        _log(f"{fs:7s} val acc={rep['accuracy']:.3f} P={rep['precision']:.3f} R={rep['recall']:.3f} "
             f"F1={rep['f1']:.3f} AUC={rep['roc_auc']:.3f} thr={rep['threshold']:.2f} "
             f"[{rep['chosen_model']}] ECE={rep['calibration_val']['ece']:.3f}")

    # Final model = best VALIDATION ROC-AUC (never chosen on dev).
    final = max(summary["sets"], key=lambda k: summary["sets"][k]["roc_auc"])
    summary["final"] = final
    shutil.copyfile(os.path.join(out_dir, f"{final}.joblib"), os.path.join(out_dir, "final.joblib"))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    _log(f"Final model (best val ROC-AUC): '{final}' -> {os.path.join(out_dir, 'final.joblib')}")
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_extract(args: argparse.Namespace) -> None:
    import torch
    from data.preprocess.hateful_memes import load_hateful_memes
    from encoders.image_encoder import ImageEncoder
    from encoders.text_encoder import TextEncoder
    from model.hm_features import extract_split

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_enc, image_enc = TextEncoder(device=device), ImageEncoder(device=device)
    for split in args.splits:
        samples = list(load_hateful_memes(args.data_dir, split))
        if split == "train" and any(s["label"] < 0 for s in samples):
            raise ValueError("train split has unlabeled rows")
        extract_split(samples, text_enc, image_enc, default_cache_path(args.data_dir, split, args.limit), args.limit)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CHEM Hateful Memes fusion classifier.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="Extract + cache features for splits.")
    p_ex.add_argument("--data-dir", default=os.path.join("data", "hateful_memes"))
    p_ex.add_argument("--splits", nargs="+", default=["train", "dev"], choices=["train", "dev", "test"])
    p_ex.add_argument("--limit", type=int, default=None, help="Use only the first N samples (smoke test).")

    p_tr = sub.add_parser("train", help="Train baselines + fusion on cached TRAIN features.")
    p_tr.add_argument("--data-dir", default=os.path.join("data", "hateful_memes"))
    p_tr.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p_tr.add_argument("--seed", type=int, default=SEED)
    p_tr.add_argument("--objective", default="balanced_accuracy", choices=["balanced_accuracy", "f1", "accuracy"])
    p_tr.add_argument("--limit", type=int, default=None)
    p_tr.add_argument("--cv-folds", type=int, default=0,
                      help="k>=2: pick model + threshold from k-fold out-of-fold predictions (more stable).")
    p_tr.add_argument("--feature-sets", nargs="+", default=None, choices=list(FEATURE_SETS),
                      help="Train only these feature sets (faster).")

    args = parser.parse_args(argv)
    if args.cmd == "extract":
        _cmd_extract(args)
    else:
        train_all(args.data_dir, args.out_dir, args.seed, args.objective, args.limit,
                  args.feature_sets, args.cv_folds)


if __name__ == "__main__":
    main()
