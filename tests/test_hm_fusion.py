"""Tests for the Hateful Memes fusion pipeline (numpy / scikit-learn only).

No torch, CLIP, transformers or dataset needed: synthetic feature caches with a
planted signal exercise splitting, scaling, threshold selection, bundle I/O and
the dev evaluator.
"""

import json

import numpy as np
import pytest

from model import hm_features as hf
from model import hm_fusion as fu

N_CONCEPT = 25
META = {
    "feature_version": hf.FEATURE_VERSION,
    "tox_labels": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
    "score_idx": [0, 1, 3, 5],
    "tox_label_to_category": {"toxic": "harassment_bullying", "identity_hate": "hate_speech"},
    "concept_prompts": [f"p{i}" for i in range(N_CONCEPT)],
    "concept_categories": ["hate_speech"] * 17 + ["neutral_scene"] * 8,
    "harmful_mask": [True] * 17 + [False] * 8,
}


def _fake_rows(n, seed, signal=1.5, dup_every=0):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < 0.4).astype(int)
    rows, ids, texts, md5 = [], [], [], []
    for i in range(n):
        txt = rng.randn(768).astype(np.float32)
        img = rng.randn(512).astype(np.float32)
        txt[0] += signal * (2 * y[i] - 1)
        img[0] += signal * (2 * y[i] - 1) * 0.5
        img /= np.linalg.norm(img)
        ctx = rng.randn(512).astype(np.float32)
        ctx /= np.linalg.norm(ctx)
        con = rng.dirichlet(np.ones(N_CONCEPT)).astype(np.float32)
        raw = {
            "txt_emb": txt[None], "img_emb": img[None], "clip_txt": ctx[None], "concept": con[None],
            "tox": rng.rand(1, 6).astype(np.float32) * (0.3 + 0.5 * y[i]),
        }
        raw.update(hf.derive_blocks(raw, META["harmful_mask"], META["score_idx"]))
        rows.append(raw)
        ids.append(f"s{seed}_{i}")
        # optional duplicated captions (confounder siblings)
        texts.append(f"caption {i // dup_every}" if dup_every else f"caption {seed}_{i}")
        md5.append(f"md5_{seed}_{i}")
    return rows, ids, texts, y.tolist(), md5


def _write_cache(path, n, seed, **kw):
    rows, ids, texts, y, md5 = _fake_rows(n, seed, **kw)
    hf._save(str(path), rows, ids, texts, y, md5, META)


def test_feature_set_dimensions():
    rows, *_ = _fake_rows(3, 0)
    blocks = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    dims = {fs: hf.build_matrix(blocks, fs).shape[1] for fs in hf.FEATURE_SETS}
    assert dims == {"text": 774, "image": 537, "concat": 1280, "policy": 1321, "fusion": 2345}
    assert blocks["agg"].shape == (3, 10) and blocks["inter"].shape == (3, 512)


def test_group_split_keeps_siblings_together_and_is_disjoint():
    rows, ids, texts, y, md5 = _fake_rows(400, 1, dup_every=3)  # triples share a caption
    tr, va = fu.split_train_val(y, texts, md5, val_frac=0.15, seed=7)
    assert not set(tr) & set(va)
    assert len(tr) + len(va) == 400
    tr_txt = {texts[i] for i in tr}
    assert not any(texts[i] in tr_txt for i in va), "caption group straddles train/val"
    assert abs(np.mean(np.asarray(y)[va]) - np.mean(y)) < 0.1  # roughly stratified


def test_split_is_deterministic():
    _, _, texts, y, md5 = _fake_rows(200, 2)
    a = fu.split_train_val(y, texts, md5, seed=5)
    b = fu.split_train_val(y, texts, md5, seed=5)
    assert all((x == z).all() for x, z in zip(a, b))


def test_best_threshold_not_degenerate():
    rng = np.random.RandomState(0)
    y = (rng.rand(500) < 0.4).astype(int)
    p = np.clip(0.55 + 0.1 * (2 * y - 1) + 0.05 * rng.randn(500), 0, 1)  # everything above 0.5
    thr = fu.best_threshold(y, p)
    rep = fu.metric_report(y, p, thr)
    assert 0.05 <= thr <= 0.95
    assert 0.2 < rep["predicted_harmful_rate"] < 0.8, "threshold collapsed to one class"


def test_metric_report_confusion_layout():
    y = [0, 0, 0, 1, 1]
    p = [0.1, 0.9, 0.2, 0.8, 0.3]
    r = fu.metric_report(y, p, 0.5)
    assert r["confusion_matrix"] == [[2, 1], [1, 1]]
    assert r["accuracy"] == pytest.approx(0.6)
    assert r["safe_recall"] == pytest.approx(2 / 3) and r["harmful_recall"] == pytest.approx(0.5)


def test_end_to_end_train_and_frozen_dev_eval(tmp_path, capsys):
    data_dir = tmp_path / "hm"
    cache = data_dir / ".cache"
    cache.mkdir(parents=True)
    _write_cache(cache / "hm_features_train.npz", 500, seed=11)
    _write_cache(cache / "hm_features_dev.npz", 120, seed=12)

    out_dir = tmp_path / "bundles"
    summary = fu.train_all(str(data_dir), str(out_dir), seed=3, feature_sets=["text", "fusion"])
    assert summary["final"] in ("text", "fusion")
    for fs in ("text", "fusion"):
        assert summary["sets"][fs]["roc_auc"] > 0.75  # planted signal must be learnable
        assert 0.05 <= summary["sets"][fs]["threshold"] <= 0.95
    assert (out_dir / "final.joblib").is_file()

    bundle = fu.FusionBundle.load(str(out_dir / "final.joblib"))
    assert bundle.info["objective"] == "balanced_accuracy"

    from evaluate import hm_eval

    res = hm_eval.evaluate(str(data_dir), str(out_dir), str(tmp_path / "out"), run_all=True)
    dev = res["dev"][summary["final"]]
    assert dev["n_samples"] == 120
    assert dev["roc_auc"] > 0.7
    assert dev["threshold"] == pytest.approx(bundle.threshold)  # threshold frozen from validation
    with open(tmp_path / "out" / "hm_error_analysis.json", encoding="utf-8") as fh:
        ea = json.load(fh)
    assert set(ea["samples"]) == {"false_positives", "false_negatives", "true_positives", "true_negatives"}
    assert "CHEM HATEFUL MEMES EVALUATION" in capsys.readouterr().out


def test_training_never_needs_or_reads_dev(tmp_path):
    data_dir = tmp_path / "hm"
    (data_dir / ".cache").mkdir(parents=True)
    _write_cache(data_dir / ".cache" / "hm_features_train.npz", 300, seed=21)
    # no dev cache exists at all: training must still succeed
    fu.train_all(str(data_dir), str(tmp_path / "b"), seed=1, feature_sets=["concat"])


def test_scaler_fit_on_train_rows_only(tmp_path):
    data_dir = tmp_path / "hm"
    (data_dir / ".cache").mkdir(parents=True)
    _write_cache(data_dir / ".cache" / "hm_features_train.npz", 300, seed=31)
    feats = hf.load_features(str(data_dir / ".cache" / "hm_features_train.npz"))
    y = feats["labels"]
    tr, va = fu.split_train_val(y, feats["texts"], feats["img_md5"], seed=1)
    bundle, _ = fu.train_feature_set(feats, y, tr, va, "concat", feats["meta"], seed=1)
    X = hf.build_matrix(feats, "concat")
    assert np.allclose(bundle.scaler.mean_, X[tr].mean(axis=0), atol=1e-5)
    assert not np.allclose(bundle.scaler.mean_, X.mean(axis=0), atol=1e-7)


def test_feature_cache_version_guard(tmp_path):
    p = tmp_path / "f.npz"
    _write_cache(p, 20, seed=1)
    with np.load(str(p), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    meta["feature_version"] = 999
    arrays["meta"] = np.asarray(json.dumps(meta))
    np.savez_compressed(str(tmp_path / "bad.npz"), **arrays)
    with pytest.raises(ValueError):
        hf.load_features(str(tmp_path / "bad.npz"))


def test_zero_shot_refuses_without_native_categories(tmp_path):
    """Hateful Memes has no category field -> the gate must report it as invalid."""
    import json as _json

    zs = pytest.importorskip("evaluate.zero_shot_eval")
    d = tmp_path / "hm"
    d.mkdir()
    (d / "dev.jsonl").write_text(_json.dumps({"id": 1, "img": "img/1.png", "text": "x", "label": 0}) + "\n")
    assert zs._has_native_categories(str(d)) is False
    (d / "dev.jsonl").write_text(
        _json.dumps({"id": 1, "img": "img/1.png", "text": "x", "label": 1, "category": "hate_speech"}) + "\n"
    )
    assert zs._has_native_categories(str(d)) is True


def test_cv_selection_uses_only_train_and_yields_valid_bundle(tmp_path):
    data_dir = tmp_path / "hm"
    (data_dir / ".cache").mkdir(parents=True)
    _write_cache(data_dir / ".cache" / "hm_features_train.npz", 300, seed=41)
    summary = fu.train_all(str(data_dir), str(tmp_path / "b"), seed=2, feature_sets=["text"], cv_folds=3)
    rep = summary["sets"]["text"]
    assert rep["cv_folds"] == 3 and rep["n_val"] == 300
    assert rep["roc_auc"] > 0.75 and 0.05 <= rep["threshold"] <= 0.95
    assert fu.FusionBundle.load(str(tmp_path / "b" / "final.joblib")).info["cv_folds"] == 3


def test_bundle_saved_from_main_module_loads_elsewhere(tmp_path):
    """`python -m model.hm_fusion train` pickles the class as __main__.FusionBundle."""
    import sys

    b = fu.FusionBundle.__new__(fu.FusionBundle)
    b.__dict__.update(scaler=None, clf=None, threshold=0.5, meta={}, info={}, feature_set="text")
    path = str(tmp_path / "m.joblib")
    orig = fu.FusionBundle.__module__
    fu.FusionBundle.__module__ = "__main__"
    sys.modules["__main__"].FusionBundle = fu.FusionBundle
    try:
        import joblib

        joblib.dump(b, path)
    finally:
        fu.FusionBundle.__module__ = orig
        delattr(sys.modules["__main__"], "FusionBundle")
    assert fu.FusionBundle.load(path).threshold == 0.5
