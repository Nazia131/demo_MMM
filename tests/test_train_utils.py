"""Tests for the data-cleaning and splitting helpers in model.train_utils."""

from model import train_utils as tu


def _s(i, text, label, img=""):
    return {"id": str(i), "img": img, "text": text, "label": label}


def test_clean_text_normalises():
    assert tu.clean_text("  Hello  \n World\x00 ") == "Hello World"
    assert tu.clean_text(None) == ""


def test_prepare_samples_drops_bad_rows(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    rows = [
        _s(1, "Same  text", 1, str(img)),
        _s(2, "same text", 1, str(img)),            # duplicate after cleaning
        _s(3, "blind", -1),                          # unlabeled
        _s(4, "gone", 0, str(tmp_path / "no.png")),  # missing image
        _s(5, "text only", 0),                       # no image is allowed
    ]
    out, st = tu.prepare_samples(rows)
    assert [r["id"] for r in out] == ["1", "5"]
    assert (st["bad_label"], st["missing_image"], st["duplicate"], st["kept"]) == (1, 1, 1, 2)


def test_groups_link_shared_text_or_image():
    rows = [_s(0, "a", 0, "x/1.png"), _s(1, "a", 1, "x/2.png"),
            _s(2, "b", 0, "x/2.png"), _s(3, "c", 1, "x/3.png")]
    g = tu.assign_groups(rows)
    assert g[0] == g[1] == g[2] and g[3] != g[0]


def test_stratified_group_split_keeps_groups_and_ratio():
    labels, groups = [], []
    for gid in range(600):
        size = 2 if gid % 5 == 0 else 1
        lab = 1 if gid % 3 == 0 else 0
        labels += [lab] * size
        groups += [gid] * size
    parts = tu.stratified_group_split(labels, groups, (0.7, 0.15, 0.15), seed=1)
    assert sorted(i for p in parts for i in p) == list(range(len(labels)))
    where = {}
    for k, p in enumerate(parts):
        for i in p:
            assert where.setdefault(groups[i], k) == k  # group never split
    overall = sum(labels) / len(labels)
    for p in parts:
        assert abs(sum(labels[i] for i in p) / len(p) - overall) < 0.05
    assert abs(len(parts[0]) / len(labels) - 0.70) < 0.03
