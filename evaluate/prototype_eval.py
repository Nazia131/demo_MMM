import sys
import os
from collections import Counter

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from api.app import CHEMPipeline
from data.preprocess.hateful_memes import load_samples


DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "hateful_memes",
)


pipeline = CHEMPipeline()
pipeline.load()

samples = list(
    load_samples(
        "dev",
        DATA_DIR,
    )
)

total = len(samples)

print("=" * 60)
print("CHEM — FALSE POSITIVE DIAGNOSTIC")
print("=" * 60)
print(f"Total samples : {total}")
print()


false_positives = []
false_negatives = []

prediction_counts = Counter()
fp_modality_counts = Counter()
fp_pattern_counts = Counter()
fp_category_counts = Counter()


for i, sample in enumerate(samples):

    try:

        with open(
            sample["img"],
            "rb",
        ) as f:
            image_bytes = f.read()

        result = pipeline.analyze(
            text=sample["text"],
            image_bytes=image_bytes,
        )

        true_label = int(
            sample["label"]
        )

        predicted_label = (
            1
            if result["verdict"] == "HARMFUL"
            else 0
        )

        prediction_counts[
            (true_label, predicted_label)
        ] += 1

        if (
            true_label == 0
            and predicted_label == 1
        ):

            fp = {
                "index": i + 1,
                "id": sample["id"],
                "text": sample["text"],
                "score": result.get(
                    "harm_score",
                    0.0,
                ),
                "modality_scores": result.get(
                    "modality_scores",
                    {},
                ),
                "harmful_modalities": result.get(
                    "harmful_modalities",
                    [],
                ),
                "patterns": result.get(
                    "triggered_patterns",
                    [],
                ),
                "category": result.get(
                    "text_pattern_categories",
                    [],
                ),
                "text_pattern_score": result.get(
                    "text_pattern_score",
                    0.0,
                ),
                "cross_modal": result.get(
                    "cross_modal_flag",
                    False,
                ),
            }

            false_positives.append(fp)

            for modality in fp[
                "harmful_modalities"
            ]:
                fp_modality_counts[
                    modality
                ] += 1

            for pattern in fp[
                "patterns"
            ]:
                fp_pattern_counts[
                    pattern
                ] += 1

            for category in fp[
                "category"
            ]:
                fp_category_counts[
                    category
                ] += 1

        elif (
            true_label == 1
            and predicted_label == 0
        ):

            false_negatives.append(
                {
                    "index": i + 1,
                    "id": sample["id"],
                    "text": sample["text"],
                    "score": result.get(
                        "harm_score",
                        0.0,
                    ),
                    "modality_scores": result.get(
                        "modality_scores",
                        {},
                    ),
                    "patterns": result.get(
                        "triggered_patterns",
                        [],
                    ),
                }
            )

        if (
            (i + 1) % 25 == 0
            or i == 0
        ):
            print(
                f"Progress: "
                f"{i + 1}/{total}"
            )

    except Exception as exc:

        print(
            f"[SKIP {i + 1}] "
            f"{exc}"
        )


print()
print("=" * 60)
print("OVERALL PREDICTIONS")
print("=" * 60)

print(
    f"True SAFE / Pred SAFE     : "
    f"{prediction_counts[(0, 0)]}"
)

print(
    f"True SAFE / Pred HARMFUL  : "
    f"{prediction_counts[(0, 1)]}"
)

print(
    f"True HARMFUL / Pred SAFE  : "
    f"{prediction_counts[(1, 0)]}"
)

print(
    f"True HARMFUL / Pred HARMFUL: "
    f"{prediction_counts[(1, 1)]}"
)


print()
print("=" * 60)
print("FALSE POSITIVE DIAGNOSTIC")
print("=" * 60)

print(
    f"False positives : "
    f"{len(false_positives)}"
)

print()

print("FP harmful modalities:")

if fp_modality_counts:

    for modality, count in (
        fp_modality_counts.most_common()
    ):
        print(
            f"  {modality:12s} : "
            f"{count}"
        )

else:

    print("  None")


print()
print("FP triggered patterns:")

if fp_pattern_counts:

    for pattern, count in (
        fp_pattern_counts.most_common()
    ):

        print(
            f"  {pattern:40s} : "
            f"{count}"
        )

else:

    print("  None")


print()
print("FP text-pattern categories:")

if fp_category_counts:

    for category, count in (
        fp_category_counts.most_common()
    ):

        print(
            f"  {category:30s} : "
            f"{count}"
        )

else:

    print("  None")


cross_modal_count = sum(
    1
    for fp in false_positives
    if fp["cross_modal"]
)

image_only_count = sum(
    1
    for fp in false_positives
    if (
        "image"
        in fp["harmful_modalities"]
        and len(
            fp["harmful_modalities"]
        ) == 1
    )
)

text_only_count = sum(
    1
    for fp in false_positives
    if (
        "text"
        in fp["harmful_modalities"]
        and len(
            fp["harmful_modalities"]
        ) == 1
    )
)

print()
print(
    f"Cross-modal FPs : "
    f"{cross_modal_count}"
)

print(
    f"Image-only FPs  : "
    f"{image_only_count}"
)

print(
    f"Text-only FPs   : "
    f"{text_only_count}"
)


print()
print("=" * 60)
print("FALSE POSITIVE EXAMPLES")
print("=" * 60)

for fp in false_positives[:30]:

    print()
    print(
        f"#{fp['index']} "
        f"ID={fp['id']}"
    )

    print(
        f"Text: "
        f"{fp['text']}"
    )

    print(
        f"Score: "
        f"{fp['score']:.4f}"
    )

    print(
        f"Modalities: "
        f"{fp['modality_scores']}"
    )

    print(
        f"Harmful modalities: "
        f"{fp['harmful_modalities']}"
    )

    print(
        f"Text pattern score: "
        f"{fp['text_pattern_score']:.4f}"
    )

    print(
        f"Patterns: "
        f"{fp['patterns']}"
    )

    print(
        f"Cross-modal: "
        f"{fp['cross_modal']}"
    )


print()
print("=" * 60)
print("FALSE NEGATIVE SUMMARY")
print("=" * 60)

print(
    f"False negatives : "
    f"{len(false_negatives)}"
)

for fn in false_negatives[:20]:

    print()
    print(
        f"#{fn['index']} "
        f"ID={fn['id']}"
    )

    print(
        f"Text: "
        f"{fn['text']}"
    )

    print(
        f"Score: "
        f"{fn['score']:.4f}"
    )

    print(
        f"Modalities: "
        f"{fn['modality_scores']}"
    )

    print(
        f"Patterns: "
        f"{fn['patterns']}"
    )


print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)