"""Feature extraction for the Hateful Memes fusion classifier.

Hateful Memes is a *binary, multimodal* task (text + image -> hateful / not).
This module turns each meme into a fixed set of plain feature blocks that a
small classifier (``model.hm_fusion``) learns from.  It deliberately produces
*features*, never verdicts: nothing here thresholds, takes a max over
modalities, or adds a hand-tuned bonus.

Feature blocks (all float32)
----------------------------
``txt_emb``   (768)  RoBERTa mean-pooled caption embedding.
``tox``       (6)    toxic-bert sigmoid probabilities (column order = ``meta.tox_labels``).
``img_emb``   (512)  CLIP ViT-B/32 image embedding (L2-normalised, NOT zero-padded).
``clip_txt``  (512)  CLIP text embedding of the caption (same space as ``img_emb``).
``concept``   (C)    CLIP softmax over the policy visual concepts (a vector, not a score).
``agg``       (10)   Policy/pattern aggregates derived from tox + concept + CLIP cosine.
``inter``     (512)  ``img_emb * clip_txt`` -- an explicit image-text interaction block.

The numeric helpers (:func:`derive_blocks`, :func:`build_matrix`, cache I/O) need
only numpy, so they are unit-testable without torch / CLIP / transformers.  The
encoder-dependent code (:func:`extract_one`, :func:`extract_split`) imports
nothing heavy at module import time.

Leakage notes
-------------
* Nothing here reads labels except to store them next to the features.
* No regex/keyword rules are used: the "pattern" features are encoder-emitted
  policy signals only (toxic-bert categories and CLIP policy concepts).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

FEATURE_VERSION = 1

#: Feature sets (name -> ordered blocks) used by baselines A-D and the fusion model.
FEATURE_SETS: Dict[str, Sequence[str]] = {
    "text": ("txt_emb", "tox"),                                         # Baseline A
    "image": ("img_emb", "concept"),                                    # Baseline B
    "concat": ("txt_emb", "img_emb"),                                   # Baseline C
    "policy": ("txt_emb", "img_emb", "tox", "concept", "agg"),          # Baseline D
    "fusion": ("txt_emb", "img_emb", "tox", "concept", "agg",           # proposed model
               "clip_txt", "inter"),
}

_RAW_BLOCKS = ("txt_emb", "tox", "img_emb", "clip_txt", "concept")
_DERIVED_BLOCKS = ("agg", "inter")


def _log(message: str) -> None:
    print(f"[CHEM][hm_features] {message}")


# --------------------------------------------------------------------------- #
# Pure-numpy derived features (testable without torch)
# --------------------------------------------------------------------------- #
def derive_blocks(
    raw: Dict[str, np.ndarray],
    harmful_mask: Sequence[bool],
    score_idx: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Compute the ``agg`` and ``inter`` blocks from the raw blocks.

    Works on a batch (leading dim N) or a single sample (leading dim 1).

    Inputs:
        raw: dict with ``tox`` (N,6), ``img_emb`` (N,512), ``clip_txt`` (N,512),
            ``concept`` (N,C).
        harmful_mask: length-C booleans, True where concept ``c`` is a harmful concept.
        score_idx: indices into the toxic-bert columns that form the text harm score.
    Outputs:
        ``{"agg": (N,10), "inter": (N,512)}``.
    """
    tox = np.asarray(raw["tox"], dtype=np.float32)
    img = np.asarray(raw["img_emb"], dtype=np.float32)
    ctx = np.asarray(raw["clip_txt"], dtype=np.float32)
    con = np.asarray(raw["concept"], dtype=np.float32)
    mask = np.asarray(harmful_mask, dtype=bool)

    harm = con[:, mask] if mask.any() else np.zeros((con.shape[0], 1), np.float32)
    safe = con[:, ~mask] if (~mask).any() else np.zeros((con.shape[0], 1), np.float32)

    max_harm = harm.max(axis=1)
    sum_harm = harm.sum(axis=1)
    max_safe = safe.max(axis=1)
    margin = max_harm - max_safe
    tox_max = tox[:, list(score_idx)].max(axis=1) if len(score_idx) else tox.max(axis=1)
    n_tox = (tox > 0.5).sum(axis=1).astype(np.float32)
    cos = (img * ctx).sum(axis=1)  # both L2-normalised -> cosine similarity
    entropy = -(con * np.log(con + 1e-9)).sum(axis=1)

    agg = np.stack(
        [max_harm, sum_harm, max_safe, margin, tox_max, n_tox, cos,
         max_harm * tox_max, cos * tox_max, entropy],
        axis=1,
    ).astype(np.float32)
    return {"agg": agg, "inter": (img * ctx).astype(np.float32)}


def build_matrix(blocks: Dict[str, np.ndarray], feature_set: str) -> np.ndarray:
    """Concatenate the blocks of ``feature_set`` into a 2-D float32 matrix."""
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
    parts = [np.asarray(blocks[name], dtype=np.float32) for name in FEATURE_SETS[feature_set]]
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------- #
# Encoder-dependent extraction
# --------------------------------------------------------------------------- #
def encoder_meta(text_encoder: Any, image_encoder: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collect the column metadata needed to interpret / re-derive features."""
    if config is None:
        from utils.config import load_policy_config

        config = load_policy_config()
    tox_labels = list(text_encoder.tox_labels)
    score_labels = config.get("toxic_bert_mapping", {}).get("text_harm_score_labels", [])
    concepts = image_encoder.concept_info
    return {
        "feature_version": FEATURE_VERSION,
        "clip_model": getattr(image_encoder, "clip_model_name", "ViT-B/32"),
        "tox_labels": tox_labels,
        "score_idx": [tox_labels.index(l) for l in score_labels if l in tox_labels],
        "tox_label_to_category": config.get("toxic_bert_mapping", {}).get("labels", {}),
        "concept_prompts": [c["prompt"] for c in concepts],
        "concept_categories": [c["category"] for c in concepts],
        "harmful_mask": [bool(c["harmful"]) for c in concepts],
    }


def extract_one(
    text: str,
    pil_image: Any,
    text_encoder: Any,
    image_encoder: Any,
    meta: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Extract all feature blocks for ONE (caption, image) pair.

    This single function is used by both batch extraction and the API, so the
    train-time and inference-time preprocessing are identical by construction.

    Outputs: dict of blocks, each with a leading dim of 1.
    """
    t = text_encoder.analyze(text)
    c = image_encoder.clip_features(pil_image, caption=text)
    raw = {
        "txt_emb": np.asarray(t["embedding"], dtype=np.float32)[None, :],
        "tox": np.asarray([t["toxic"][l] for l in meta["tox_labels"]], dtype=np.float32)[None, :],
        "img_emb": np.asarray(c["image_embedding"], dtype=np.float32)[None, :],
        "clip_txt": np.asarray(c["text_embedding"], dtype=np.float32)[None, :],
        "concept": np.asarray(c["concept_probs"], dtype=np.float32)[None, :],
    }
    raw.update(derive_blocks(raw, meta["harmful_mask"], meta["score_idx"]))
    return raw


def _md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def extract_split(
    samples: List[Dict[str, Any]],
    text_encoder: Any,
    image_encoder: Any,
    cache_path: str,
    limit: Optional[int] = None,
    checkpoint_every: int = 500,
) -> Dict[str, Any]:
    """Extract (or load from cache) features for a list of samples.

    Resumable: a ``.partial.npz`` is written every ``checkpoint_every`` samples
    so an interrupted CPU run does not start over.

    Inputs:
        samples: ``{"id","img","text","label"}`` dicts from the loader.
        cache_path: ``.npz`` file to read/write.
        limit: Optional deterministic prefix length (smoke tests).
    Outputs:
        dict with blocks plus ``ids``, ``texts``, ``labels``, ``img_md5``, ``meta``.
    """
    from PIL import Image

    if limit:
        samples = samples[:limit]
    if os.path.isfile(cache_path):
        cached = load_features(cache_path)
        if list(cached["ids"]) == [str(s["id"]) for s in samples]:
            _log(f"Loaded cached features: {cache_path} ({len(cached['ids'])} samples)")
            return cached
        _log("Cache id list differs from requested samples; re-extracting.")

    meta = encoder_meta(text_encoder, image_encoder)
    partial_path = cache_path + ".partial.npz"
    rows: List[Dict[str, np.ndarray]] = []
    ids: List[str] = []
    texts: List[str] = []
    labels: List[int] = []
    md5s: List[str] = []
    done = set()

    if os.path.isfile(partial_path):
        part = load_features(partial_path)
        want = [str(s["id"]) for s in samples]
        if list(part["ids"]) == want[: len(part["ids"])]:
            n = len(part["ids"])
            rows = [{k: part[k][i : i + 1] for k in _RAW_BLOCKS + _DERIVED_BLOCKS} for i in range(n)]
            ids, texts = list(part["ids"]), list(part["texts"])
            labels, md5s = [int(v) for v in part["labels"]], list(part["img_md5"])
            done = set(ids)
            _log(f"Resuming from {n} samples in {partial_path}")

    skipped = 0
    for i, sample in enumerate(samples):
        sid = str(sample["id"])
        if sid in done:
            continue
        added = False
        try:
            with Image.open(sample["img"]) as im:
                pil = im.convert("RGB")
            rows.append(extract_one(sample.get("text", "") or "", pil, text_encoder, image_encoder, meta))
            ids.append(sid)
            texts.append(sample.get("text", "") or "")
            labels.append(int(sample.get("label", -1)))
            md5s.append(_md5(sample["img"]))
            added = True
        except Exception as exc:
            skipped += 1
            _log(f"Skipping sample {sid}: {exc}")
        if (i + 1) % 100 == 0 or (i + 1) == len(samples):
            _log(f"extracted {i + 1}/{len(samples)}")
        if added and checkpoint_every and len(ids) % checkpoint_every == 0:
            _save(partial_path, rows, ids, texts, labels, md5s, meta)

    if not rows:
        raise RuntimeError("No features extracted; check image paths / encoders.")
    out = _save(cache_path, rows, ids, texts, labels, md5s, meta)
    if os.path.isfile(partial_path):
        os.remove(partial_path)
    _log(f"Saved features -> {cache_path} ({len(ids)} samples, {skipped} skipped)")
    return out


def _save(path, rows, ids, texts, labels, md5s, meta) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blocks = {k: np.concatenate([r[k] for r in rows], axis=0) for k in _RAW_BLOCKS + _DERIVED_BLOCKS}
    arrays = dict(
        blocks,
        ids=np.asarray(ids, dtype=str),
        texts=np.asarray(texts, dtype=str),
        labels=np.asarray(labels, dtype=np.int64),
        img_md5=np.asarray(md5s, dtype=str),
        meta=np.asarray(json.dumps(meta)),
    )
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return load_features(path)


def load_features(path: str) -> Dict[str, Any]:
    """Load a feature cache written by :func:`extract_split` (no pickle needed)."""
    with np.load(path, allow_pickle=False) as z:
        out: Dict[str, Any] = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    if out["meta"].get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"{path}: feature_version {out['meta'].get('feature_version')} != {FEATURE_VERSION}; re-extract."
        )
    return out


def default_cache_path(data_dir: str, split: str, limit: Optional[int] = None) -> str:
    """``<data_dir>/.cache/hm_features_<split>[_<limit>].npz`` (git-ignored)."""
    suffix = f"_{limit}" if limit else ""
    return os.path.join(data_dir, ".cache", f"hm_features_{split}{suffix}.npz")
