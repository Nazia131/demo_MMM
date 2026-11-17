"""End-to-end accuracy check against a running CHEM server.

Sends every row of a CSV of labeled cases to ``POST /analyze`` and reports
accuracy, precision, recall, F1, a confusion matrix and every miss (with the
scores and the OCR text / transcript, so you can see *why* it was missed).

CSV columns (header row required; leave a cell empty if the modality is unused)::

    id,text,image,audio,expected,note
    t1,I will kill you,,,HARMFUL,direct threat
    t2,nice weather today,,,SAFE,benign
    m1,,memes/knife.avif,,HARMFUL,knife image
    a1,,,clips/threat.wav,HARMFUL,spoken threat

``image`` / ``audio`` paths are relative to the CSV file.  ``expected`` is
``HARMFUL`` or ``SAFE``.

Usage (server must already be running)::

    python -m evaluate.api_accuracy evaluate/cases.csv
    python -m evaluate.api_accuracy evaluate/cases.csv --url http://127.0.0.1:8000 --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional

LABELS = ("HARMFUL", "SAFE")


def _read_cases(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [
            {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            for row in csv.DictReader(handle)
        ]
    cases = []
    for i, row in enumerate(rows, start=2):
        if not any(row.get(k) for k in ("text", "image", "audio")):
            continue  # blank line
        expected = row.get("expected", "").upper()
        if expected not in LABELS:
            raise SystemExit(
                f"{path} line {i}: 'expected' must be HARMFUL or SAFE, got {expected!r}"
            )
        row["expected"] = expected
        row.setdefault("id", f"row{i}")
        cases.append(row)
    return cases


def _call(url: str, case: Dict[str, str], base_dir: str, timeout: float) -> Dict[str, Any]:
    import requests

    data = {"text": case["text"]} if case.get("text") else {}
    files: Dict[str, Any] = {}
    handles = []
    try:
        for field in ("image", "audio"):
            rel = case.get(field)
            if rel:
                full = os.path.join(base_dir, rel)
                if not os.path.isfile(full):
                    raise FileNotFoundError(full)
                fh = open(full, "rb")
                handles.append(fh)
                files[field] = (os.path.basename(full), fh)
        resp = requests.post(f"{url}/analyze", data=data, files=files or None, timeout=timeout)
        body = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(body.get("error") or resp.text[:200])
        return body
    finally:
        for fh in handles:
            fh.close()


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    scored = [r for r in rows if r["predicted"] in LABELS]
    tp = sum(r["expected"] == "HARMFUL" and r["predicted"] == "HARMFUL" for r in scored)
    tn = sum(r["expected"] == "SAFE" and r["predicted"] == "SAFE" for r in scored)
    fp = sum(r["expected"] == "SAFE" and r["predicted"] == "HARMFUL" for r in scored)
    fn = sum(r["expected"] == "HARMFUL" and r["predicted"] == "SAFE" for r in scored)
    n = len(scored)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
    }


def _kind(case: Dict[str, str]) -> str:
    return "+".join(k for k in ("text", "image", "audio") if case.get(k)) or "none"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", help="CSV of labeled cases")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL")
    ap.add_argument("--out", help="write per-case results to this CSV")
    ap.add_argument("--timeout", type=float, default=180.0, help="seconds per request")
    args = ap.parse_args(argv)

    base_dir = os.path.dirname(os.path.abspath(args.cases))
    cases = _read_cases(args.cases)
    if not cases:
        raise SystemExit("No cases found in the CSV.")

    try:
        import requests

        health = requests.get(f"{args.url}/health", timeout=5).json()
    except Exception as exc:
        raise SystemExit(f"Cannot reach {args.url}/health ({exc}). Is the server running?")
    if not health.get("models_loaded"):
        raise SystemExit(f"Server is up but models are not loaded: {health.get('error')}")
    print(f"Server OK  ocr={health.get('ocr_available')}  audio={health.get('audio_available')}\n")

    rows: List[Dict[str, Any]] = []
    for case in cases:
        row: Dict[str, Any] = {
            "id": case["id"], "kind": _kind(case), "expected": case["expected"],
            "note": case.get("note", ""), "predicted": "ERROR", "score": "",
            "modality_scores": "", "ocr_text": "", "transcript": "", "warnings": "", "error": "",
        }
        try:
            res = _call(args.url, case, base_dir, args.timeout)
            row.update(
                predicted=res.get("verdict", "ERROR"),
                score=res.get("harm_score"),
                modality_scores=res.get("modality_scores"),
                ocr_text=res.get("ocr_text", ""),
                transcript=res.get("transcript", ""),
                warnings=" | ".join(res.get("warnings", [])),
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        ok = "ok  " if row["predicted"] == row["expected"] else "MISS"
        print(f"[{ok}] {row['id']:<10} {row['kind']:<12} expected={row['expected']:<8} "
              f"got={row['predicted']:<8} score={row['score']}")
        rows.append(row)

    m = _metrics(rows)
    errors = [r for r in rows if r["predicted"] == "ERROR"]
    print("\n" + "=" * 60)
    print(f"cases scored: {m['n']}  (errors: {len(errors)})")
    print(f"accuracy : {m['accuracy']:.1%}")
    print(f"precision: {m['precision']:.1%}   recall: {m['recall']:.1%}   F1: {m['f1']:.3f}")
    print("\nconfusion (rows = expected, cols = predicted)")
    print("              HARMFUL   SAFE")
    print(f"  HARMFUL     {m['tp']:>7}  {m['fn']:>5}   <- misses here are false negatives")
    print(f"  SAFE        {m['fp']:>7}  {m['tn']:>5}   <- misses here are false positives")

    print("\naccuracy by input type")
    for kind in sorted({r["kind"] for r in rows}):
        sub = _metrics([r for r in rows if r["kind"] == kind])
        print(f"  {kind:<14} {sub['accuracy']:>6.1%}  ({sub['tp'] + sub['tn']}/{sub['n']})")

    misses = [r for r in rows if r["predicted"] in LABELS and r["predicted"] != r["expected"]]
    if misses or errors:
        print("\nMISSES / ERRORS")
        for r in misses + errors:
            print(f"- {r['id']} ({r['kind']}) expected {r['expected']}, got {r['predicted']}, score={r['score']}")
            for key in ("modality_scores", "ocr_text", "transcript", "warnings", "error", "note"):
                if r[key]:
                    print(f"    {key}: {r[key]}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nper-case results written to {args.out}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
