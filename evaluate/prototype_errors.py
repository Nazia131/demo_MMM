import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from api.app import CHEMPipeline
from data.preprocess.hateful_memes import load_samples

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "hateful_memes")
N = 100

pipeline = CHEMPipeline()
pipeline.load()

errors = []

samples = load_samples("dev", DATA_DIR)

for i, sample in enumerate(samples):
    if i >= N:
        break

    try:
        with open(sample["img"], "rb") as f:
            image_bytes = f.read()

        result = pipeline.analyze(
            text=sample["text"],
            image_bytes=image_bytes,
        )

        true_label = int(sample["label"])
        pred_label = 1 if result["verdict"] == "HARMFUL" else 0

        if true_label != pred_label:
            errors.append({
                "index": i + 1,
                "id": sample["id"],
                "true": true_label,
                "pred": pred_label,
                "score": result["harm_score"],
                "text": sample["text"],
                "patterns": result.get("triggered_patterns", []),
                "modalities": result.get("harmful_modalities", []),
                "matched_pattern": result.get("matched_pattern"),
            })

    except Exception as exc:
        print(f"[SKIP {i+1}] {exc}")

print("\n" + "=" * 80)
print("CHEM PROTOTYPE — MISCLASSIFIED SAMPLES")
print("=" * 80)

print(f"Total errors: {len(errors)}")

false_positives = [e for e in errors if e["true"] == 0 and e["pred"] == 1]
false_negatives = [e for e in errors if e["true"] == 1 and e["pred"] == 0]

print(f"False positives: {len(false_positives)}")
print(f"False negatives: {len(false_negatives)}")

print("\n" + "=" * 80)
print("FALSE POSITIVES")
print("(Dataset says SAFE, CHEM says HARMFUL)")
print("=" * 80)

for e in false_positives:
    print(f"\nSample #{e['index']}  ID={e['id']}")
    print(f"True: {e['true']}  Pred: {e['pred']}  Score: {e['score']:.4f}")
    print(f"Text: {e['text']}")
    print(f"Modalities: {e['modalities']}")
    print(f"Matched pattern: {e['matched_pattern']}")
    print(f"Patterns: {e['patterns']}")

print("\n" + "=" * 80)
print("FALSE NEGATIVES")
print("(Dataset says HARMFUL, CHEM says SAFE)")
print("=" * 80)

for e in false_negatives:
    print(f"\nSample #{e['index']}  ID={e['id']}")
    print(f"True: {e['true']}  Pred: {e['pred']}  Score: {e['score']:.4f}")
    print(f"Text: {e['text']}")
    print(f"Modalities: {e['modalities']}")
    print(f"Matched pattern: {e['matched_pattern']}")
    print(f"Patterns: {e['patterns']}")

print("\n" + "=" * 80)
