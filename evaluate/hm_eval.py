"""Reproducible Hateful Memes evaluation of the frozen CHEM fusion bundles.

This script NEVER fits anything.  It loads bundles produced by
``python -m model.hm_fusion train`` (scaler, classifier and threshold were all
fixed using the official TRAIN split only) and scores the untouched ``dev``
split from cached features.

    python -m evaluate.hm_eval --data-dir data/hateful_memes
    python -m evaluate.hm_eval --data-dir data/hateful_memes --all

``--all`` additionally reports baselines A-D for the comparison table.  The
final model is ALWAYS the one selected on validation ROC-AUC (recorded in
``summary.json``); dev numbers are never used to pick a model or a threshold.

Outputs (default ``evaluate_outputs/``, git-ignored)::

    hm_eval_report.json      metrics for every evaluated bundle + leakage check
    hm_error_analysis.json   20 FP / 20 FN / 20 TP / 20 TN with feature scores
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from model.hm_features import default_cache_path, load_features
from model.hm_fusion import DEFAULT_OUT_DIR, FusionBundle, metric_report, _norm_text

_BASELINE_NAMES = {
    "text": "A  text only",
    "image": "B  image only",
    "concat": "C  text+image concat",
    "policy": "D  C + policy features",
    "fusion": "   fusion (interaction)",
}


def _pct(x: float) -> str:
    return "  n/a " if x != x else f"{100.0 * x:6.2f}%"


def print_report(title: str, r: Dict[str, Any]) -> None:
    """Print one report in the standard CHEM format."""
    tn, fp = r["confusion_matrix"][0]
    fn, tp = r["confusion_matrix"][1]
    line = "=" * 60
    print(line)
    print(title)
    print(line)
    print(f"Samples evaluated : {r['n_samples']}   (SAFE {r['n_safe']} / HARMFUL {r['n_harmful']})")
    print(f"Threshold (frozen): {r['threshold']:.2f}   predicted-harmful rate {r['predicted_harmful_rate']:.1%}")
    print()
    print(f"Accuracy           : {_pct(r['accuracy'])}")
    print(f"Precision          : {_pct(r['precision'])}")
    print(f"Recall             : {_pct(r['recall'])}")
    print(f"F1                 : {_pct(r['f1'])}")
    print(f"ROC-AUC            : {_pct(r['roc_auc'])}")
    print(f"PR-AUC             : {_pct(r['pr_auc'])}")
    print()
    print(f"SAFE precision     : {_pct(r['safe_precision'])}   SAFE recall     : {_pct(r['safe_recall'])}")
    print(f"HARMFUL precision  : {_pct(r['harmful_precision'])}   HARMFUL recall  : {_pct(r['harmful_recall'])}")
    print()
    print("Confusion Matrix:")
    print(f"[[{tn:4d} {fp:4d}]")
    print(f" [{fn:4d} {tp:4d}]]")
    print(line)


def leakage_check(dev: Dict[str, Any], train_path: Optional[str]) -> Dict[str, Any]:
    """Count exact caption / image overlaps between train and dev (report only)."""
    out: Dict[str, Any] = {"dev_rows": int(len(dev["ids"]))}
    if not train_path or not os.path.isfile(train_path):
        out["note"] = "train feature cache not found; overlap not checked"
        return out
    tr = load_features(train_path)
    tr_txt = {_norm_text(t) for t in tr["texts"] if _norm_text(t)}
    tr_md5 = set(tr["img_md5"])
    out["dev_images_identical_to_train"] = int(sum(m in tr_md5 for m in dev["img_md5"]))
    out["dev_captions_identical_to_train"] = int(sum(_norm_text(t) in tr_txt for t in dev["texts"]))
    out["dev_ids_also_in_train"] = int(len(set(dev["ids"]) & set(tr["ids"])))
    return out


def _sample_row(i: int, dev: Dict[str, Any], prob: float, pred: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    meta = dev["meta"]
    tox = dev["tox"][i]
    conc = dev["concept"][i]
    mask = np.asarray(meta["harmful_mask"], dtype=bool)
    text_score = float(tox[meta["score_idx"]].max()) if meta["score_idx"] else float(tox.max())
    harm_conc = np.where(mask, conc, 0.0)
    top = int(harm_conc.argmax())
    image_score = float(harm_conc[top])

    patterns: List[str] = []
    for label, p in zip(meta["tox_labels"], tox):
        cat = meta["tox_label_to_category"].get(label)
        if p >= cfg["tox_threshold"] and cat and cat not in patterns:
            patterns.append(f"text:{cat}")
    if image_score >= cfg["image_min_conf"]:
        patterns.append(f"image:{meta['concept_categories'][top]}")
    return {
        "id": str(dev["ids"][i]),
        "text": str(dev["texts"][i]),
        "true_label": int(dev["labels"][i]),
        "pred_label": int(pred),
        "probability": round(float(prob), 4),
        "text_score": round(text_score, 4),
        "image_score": round(image_score, 4),
        "fusion_score": round(float(prob), 4),
        "pattern_matches": patterns,
        "clip_image_text_cosine": round(float(dev["agg"][i][6]), 4),
        "bucket": ("text_flag" if text_score >= 0.5 else "text_benign")
                  + "+" + ("image_flag" if image_score >= 0.5 else "image_benign"),
    }


def error_analysis(dev: Dict[str, Any], prob: np.ndarray, thr: float, per_group: int = 20) -> Dict[str, Any]:
    """Pick the most confident 20 FP / FN / TP / TN and tabulate score buckets."""
    cfg = {"tox_threshold": 0.5, "image_min_conf": 0.25}
    try:
        from utils.config import get_coefficient, load_policy_config

        cfg["image_min_conf"] = float(get_coefficient("image_harmful_min_confidence"))
        cfg["tox_threshold"] = float(load_policy_config()["toxic_bert_mapping"]["harmful_signal_threshold"])
    except Exception:
        pass

    y = dev["labels"].astype(int)
    pred = (prob >= thr).astype(int)
    groups = {
        "false_positives": (np.flatnonzero((y == 0) & (pred == 1)), True),   # most confident wrong first
        "false_negatives": (np.flatnonzero((y == 1) & (pred == 0)), False),
        "true_positives": (np.flatnonzero((y == 1) & (pred == 1)), True),
        "true_negatives": (np.flatnonzero((y == 0) & (pred == 0)), False),
    }
    out: Dict[str, Any] = {"threshold": float(thr), "samples": {}, "bucket_counts": {}}
    for name, (idx, descending) in groups.items():
        order = idx[np.argsort(-prob[idx] if descending else prob[idx], kind="stable")]
        out["samples"][name] = [_sample_row(int(i), dev, prob[i], pred[i], cfg) for i in order[:per_group]]
        buckets: Dict[str, int] = {}
        for i in idx:
            b = _sample_row(int(i), dev, prob[i], pred[i], cfg)["bucket"]
            buckets[b] = buckets.get(b, 0) + 1
        out["bucket_counts"][name] = {"total": int(len(idx)), **dict(sorted(buckets.items()))}
    return out


def evaluate(data_dir: str, bundle_dir: str, out_dir: str, run_all: bool) -> Dict[str, Any]:
    dev_path = default_cache_path(data_dir, "dev")
    if not os.path.isfile(dev_path):
        raise FileNotFoundError(f"{dev_path} not found. Run: python -m model.hm_fusion extract --splits dev")
    dev = load_features(dev_path)
    if (dev["labels"] < 0).any():
        raise ValueError("dev split has unlabeled rows; cannot compute metrics.")

    with open(os.path.join(bundle_dir, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    final = summary["final"]
    names = list(summary["sets"]) if run_all else [final]
    if final not in names:
        names.append(final)

    leak = leakage_check(dev, default_cache_path(data_dir, "train"))
    print("[CHEM][hm_eval] leakage check (train vs dev):", json.dumps(leak))

    results: Dict[str, Any] = {"final_model": final, "leakage_check": leak, "dev": {}}
    probs: Dict[str, np.ndarray] = {}
    for name in names:
        bundle = FusionBundle.load(os.path.join(bundle_dir, f"{name}.joblib"))
        if bundle.meta["feature_version"] != dev["meta"]["feature_version"]:
            raise ValueError(f"{name}: feature_version mismatch between bundle and dev cache.")
        p = bundle.predict_proba(dev)
        probs[name] = p
        results["dev"][name] = metric_report(dev["labels"], p, bundle.threshold)

    marker = "  <-- FINAL (selected on validation)"
    if run_all:
        print("\nDEV comparison (frozen thresholds from validation; no selection on dev)")
        print(f"{'model':26s} {'acc':>7s} {'prec':>7s} {'rec':>7s} {'F1':>7s} {'ROC':>7s} {'PR':>7s}")
        for name in names:
            r = results["dev"][name]
            print(f"{_BASELINE_NAMES.get(name, name):26s} {_pct(r['accuracy'])} {_pct(r['precision'])} "
                  f"{_pct(r['recall'])} {_pct(r['f1'])} {_pct(r['roc_auc'])} {_pct(r['pr_auc'])}"
                  f"{marker if name == final else ''}")
        print()
    print_report("CHEM HATEFUL MEMES EVALUATION (dev, untouched)", results["dev"][final])

    results["error_analysis_file"] = os.path.join(out_dir, "hm_error_analysis.json")
    ea = error_analysis(dev, probs[final], float(results["dev"][final]["threshold"]))
    os.makedirs(out_dir, exist_ok=True)
    with open(results["error_analysis_file"], "w", encoding="utf-8") as fh:
        json.dump(ea, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "hm_eval_report.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\nError buckets (text/image flag = encoder score >= 0.5):")
    for group, counts in ea["bucket_counts"].items():
        print(f"  {group:16s} {counts}")
    print(f"\nWrote {out_dir}{os.sep}hm_eval_report.json and hm_error_analysis.json")
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen CHEM fusion bundles on Hateful Memes dev.")
    parser.add_argument("--data-dir", default=os.path.join("data", "hateful_memes"))
    parser.add_argument("--bundle-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-dir", default="evaluate_outputs")
    parser.add_argument("--all", action="store_true", help="Also report baselines A-D.")
    args = parser.parse_args(argv)
    evaluate(args.data_dir, args.bundle_dir, args.out_dir, args.all)


if __name__ == "__main__":
    main()
