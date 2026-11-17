"""Small, testable helpers for ``model.train``.

* class weights for an imbalanced binary label set,
* decision-threshold selection on a validation set,
* metrics at an arbitrary threshold,
* an on-disk cache of built graphs, keyed by a fingerprint of the samples *and*
  of the source files that shape a graph, so editing an encoder or the graph
  builder automatically invalidates stale graphs.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# Files whose contents change what a built graph looks like.
_GRAPH_SHAPING_FILES = (
    "encoders/text_encoder.py",
    "encoders/image_encoder.py",
    "graph/graph_builder.py",
    "utils/embeddings.py",
    "policy/policy_config.json",
)

# Bump when the graphs' stored attributes change (e.g. new ``group_id`` field).
_CACHE_VERSION = "3"


# ------------------------------------------------------------------ weights
def class_weights(labels: Iterable[int]) -> List[float]:
    """Inverse-frequency weights ``n / (2 * n_c)`` for classes 0 and 1.

    A class with no examples gets weight 1.0 so the loss stays defined.
    """
    labels = [int(l) for l in labels]
    n = len(labels)
    counts = [labels.count(0), labels.count(1)]
    return [n / (2.0 * c) if c else 1.0 for c in counts]


# ---------------------------------------------------------------- threshold
def metrics_at_threshold(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    """accuracy / precision / recall / F1 / AUC with ``p >= threshold`` => 1."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0
    except Exception:
        auc = 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
    }


def best_threshold(
    y_true: Sequence[int], y_prob: Sequence[float], metric: str = "f1"
) -> float:
    """Grid-search the threshold (0.05..0.95, step 0.01) maximising ``metric``.

    Meant to be called on VALIDATION predictions only; ties keep the threshold
    closest to 0.5 so the result is stable.
    """
    if metric not in ("f1", "accuracy", "balanced_accuracy"):
        raise ValueError("metric must be 'f1', 'accuracy' or 'balanced_accuracy'")
    import numpy as np

    y_arr = np.asarray(list(y_true), dtype=int)
    best_t, best_v = 0.5, -1.0
    for i in range(5, 96):
        t = i / 100.0
        if metric == "balanced_accuracy":
            # Mean of SAFE recall and HARMFUL recall: cannot be gamed by predicting
            # one class and does not depend on the split's class prior.
            pred = (np.asarray(list(y_prob), dtype=float) >= t).astype(int)
            rec_pos = float(((pred == 1) & (y_arr == 1)).sum()) / max(int((y_arr == 1).sum()), 1)
            rec_neg = float(((pred == 0) & (y_arr == 0)).sum()) / max(int((y_arr == 0).sum()), 1)
            v = 0.5 * (rec_pos + rec_neg)
        else:
            v = metrics_at_threshold(y_true, y_prob, t)[metric]
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_v = t, v
    return best_t


# -------------------------------------------------------------------- cache
def graph_cache_key(samples: Sequence[Dict[str, Any]], root: str = _ROOT) -> str:
    """Fingerprint of the samples plus the files that shape a graph."""
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    for s in samples:
        h.update(repr((s.get("id"), s.get("img"), s.get("text"), s.get("label"))).encode("utf-8"))
    for rel in _GRAPH_SHAPING_FILES:
        path = os.path.join(root, rel)
        h.update(rel.encode())
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"graphs_{key}.pt")


def load_graph_cache(cache_dir: Optional[str], key: str) -> Optional[List[Any]]:
    """Return cached graphs for ``key`` or ``None`` (missing / unreadable)."""
    if not cache_dir:
        return None
    path = cache_path(cache_dir, key)
    if not os.path.isfile(path):
        return None
    import torch

    try:
        # Files in this folder are written by save_graph_cache() on this machine;
        # PyG Data objects need full unpickling, so never point this at a file
        # you did not create.
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[CHEM][train] cache unreadable ({exc}); rebuilding.")
        return None


def save_graph_cache(cache_dir: Optional[str], key: str, graphs: List[Any]) -> Optional[str]:
    if not cache_dir:
        return None
    import torch

    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, key)
    tmp = path + ".tmp"
    torch.save(graphs, tmp)
    os.replace(tmp, path)  # never leave a half-written cache behind
    return path


def subsample(samples: List[Dict[str, Any]], max_samples: Optional[int], seed: int = 42) -> List[Dict[str, Any]]:
    """Deterministic random subset (all samples when ``max_samples`` is falsy)."""
    if not max_samples or max_samples >= len(samples):
        return samples
    import random

    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    return [samples[i] for i in sorted(idx[:max_samples])]


# ----------------------------------------------------------------- cleaning
def clean_text(text: Any) -> str:
    """Normalise caption text: NFKC, drop control chars, collapse whitespace."""
    s = unicodedata.normalize("NFKC", "" if text is None else str(text))
    s = "".join(ch for ch in s if ch in "\n\t " or unicodedata.category(ch)[0] != "C")
    return re.sub(r"\s+", " ", s).strip()


def _text_key(text: str) -> str:
    return clean_text(text).casefold()


def _img_key(img: Any) -> str:
    return os.path.basename(img).casefold() if isinstance(img, str) and img else ""


def prepare_samples(samples: Sequence[Dict[str, Any]]) -> tuple:
    """Clean raw samples; returns ``(clean_samples, stats)``.

    * label must be 0/1 (drops unlabeled rows, e.g. ``-1`` from blind splits),
    * a non-empty ``img`` path that does not exist is dropped (otherwise the
      meme silently becomes text-only while keeping its image-dependent label),
    * captions are cleaned with :func:`clean_text`,
    * exact duplicates (same cleaned text + image file name) keep the first copy.
    """
    stats = {"input": len(samples), "bad_label": 0, "missing_image": 0, "duplicate": 0}
    seen, out = set(), []
    for s in samples:
        try:
            label = int(s.get("label", -1))
        except (TypeError, ValueError):
            label = -1
        if label not in (0, 1):
            stats["bad_label"] += 1
            continue
        img = s.get("img")
        if isinstance(img, str) and img and not os.path.isfile(img):
            stats["missing_image"] += 1
            continue
        text = clean_text(s.get("text"))
        key = (_text_key(text), _img_key(img))
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        out.append({**s, "text": text, "label": label})
    stats["kept"] = len(out)
    return out, stats


def assign_groups(samples: Sequence[Dict[str, Any]]) -> List[int]:
    """Group id per sample: samples sharing a caption OR an image are linked.

    Hateful Memes ships "confounder" pairs (same text/different image and vice
    versa); a random split would put near-twins in both train and test and
    inflate scores, so splits are done per group.
    """
    parent = list(range(len(samples)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: Dict[Any, int] = {}
    for i, s in enumerate(samples):
        for key in (("t", _text_key(s.get("text", ""))), ("i", _img_key(s.get("img")))):
            if not key[1]:
                continue
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    return [find(i) for i in range(len(samples))]


def stratified_group_split(
    labels: Sequence[int],
    groups: Sequence[int],
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> List[List[int]]:
    """Split indices into ``len(fractions)`` parts keeping each group intact and
    the harmful/safe ratio of every part close to the overall ratio."""
    import random

    n_split = len(fractions)
    by_group: Dict[int, List[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    items = list(by_group.values())
    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=len, reverse=True)  # big groups first, ties stay shuffled

    total = len(labels)
    total_pos = sum(int(l) == 1 for l in labels)
    tgt_n = [f * total for f in fractions]
    tgt_pos = [f * total_pos for f in fractions]
    tgt_neg = [a - b for a, b in zip(tgt_n, tgt_pos)]
    cur_pos, cur_neg = [0] * n_split, [0] * n_split
    parts: List[List[int]] = [[] for _ in range(n_split)]

    for idxs in items:
        pos = sum(int(labels[i]) == 1 for i in idxs)
        neg = len(idxs) - pos
        best, best_score = 0, None
        for s in range(n_split):
            need_pos = (tgt_pos[s] - cur_pos[s]) / max(tgt_pos[s], 1.0)
            need_neg = (tgt_neg[s] - cur_neg[s]) / max(tgt_neg[s], 1.0)
            score = (pos * need_pos + neg * need_neg) / len(idxs)
            if best_score is None or score > best_score:
                best, best_score = s, score
        parts[best].extend(idxs)
        cur_pos[best] += pos
        cur_neg[best] += neg
    return [sorted(p) for p in parts]