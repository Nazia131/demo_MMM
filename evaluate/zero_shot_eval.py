"""
Zero-shot generalisation evaluation for CHEM.

Hold out one complete harm category from training and evaluate on samples
belonging to that unseen category.

The headline metric is zero-shot F1.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluate.metrics import compute_metrics, pretty_print
from utils.config import get_coefficient, get_harm_categories


# ---------------------------------------------------------------------------
# Lazy dependency access
# ---------------------------------------------------------------------------

def _try_import_torch():
    """Import torch lazily."""
    try:
        import torch
        return torch
    except Exception as exc:
        print(f"[CHEM][zero_shot] torch is unavailable: {exc}")
        return None


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------

def _resolve_categories(
    held_out_category: str,
) -> Tuple[List[str], List[str]]:
    """Split configured categories into train and held-out categories."""

    all_categories = list(get_harm_categories().keys())

    if held_out_category not in all_categories:
        raise ValueError(
            f"Unknown held-out category '{held_out_category}'. "
            f"Valid categories: {all_categories}"
        )

    train_categories = [
        category
        for category in all_categories
        if category != held_out_category
    ]

    return train_categories, [held_out_category]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_model_excluding(
    train_categories: Sequence[str],
    train_fn: Optional[Callable[..., Any]],
    epochs: int,
) -> Optional[Any]:
    """Train a CHEM GAT without the held-out category."""

    if train_fn is None:
        try:
            from model import train as train_module

            # FIXED: this line must be inside the try block.
            train_fn = getattr(train_module, "train", None)

        except Exception as exc:
            print(
                f"[CHEM][zero_shot] model.train unavailable: {exc}"
            )
            train_fn = None

    if train_fn is None:
        print(
            "[CHEM][zero_shot] No training entry point found. "
            "Supply a trained model and test_loader instead."
        )
        return None

    print(
        f"[CHEM][zero_shot] Training on "
        f"{len(train_categories)} categories:"
    )
    for category in train_categories:
        print(f"  - {category}")

    try:
        return train_fn(
            include_categories=list(train_categories),
            epochs=epochs,
        )

    except TypeError:
        print(
            "[CHEM][zero_shot] train() does not support "
            "include_categories. Falling back to normal training."
        )

        try:
            return train_fn(epochs=epochs)
        except TypeError:
            return train_fn()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _predict_probabilities(
    model: Any,
    test_loader: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model and return true labels + harmful probabilities."""

    torch = _try_import_torch()

    if torch is None:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    if model is None or test_loader is None:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    model.eval()

    all_true: List[int] = []
    all_prob: List[float] = []

    with torch.no_grad():

        for batch in test_loader:

            # Support CPU/GPU models.
            try:
                device = next(model.parameters()).device
                batch = batch.to(device)
            except Exception:
                pass

            log_probs = model(
                batch.x,
                batch.edge_index,
                batch.batch,
            )

            probs = torch.exp(log_probs)

            harmful_prob = probs[:, 1]

            all_prob.extend(
                harmful_prob.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            all_true.extend(
                batch.y.detach()
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        np.asarray(all_true, dtype=int),
        np.asarray(all_prob, dtype=float),
    )


# ---------------------------------------------------------------------------
# Main zero-shot orchestration
# ---------------------------------------------------------------------------

def run_zero_shot_eval(
    held_out_category: str,
    model: Optional[Any] = None,
    test_loader: Optional[Any] = None,
    train_fn: Optional[Callable[..., Any]] = None,
    epochs: int = 100,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run leave-one-category-out zero-shot evaluation.
    """

    print("=" * 70)
    print(
        f"[CHEM][zero_shot] "
        f"Zero-shot evaluation — holding out '{held_out_category}'"
    )
    print("=" * 70)

    train_categories, held_out = _resolve_categories(
        held_out_category
    )

    print(
        f"[CHEM][zero_shot] Train categories : "
        f"{train_categories}"
    )
    print(
        f"[CHEM][zero_shot] Held-out category: "
        f"{held_out}"
    )

    result: Dict[str, Any] = {
        "held_out_category": held_out_category,
        "train_categories": train_categories,
        "metrics": None,
        "zero_shot_f1": None,
        "trained": False,
        "evaluated": False,
    }

    # ---------------------------------------------------------------
    # Step 1 — obtain model
    # ---------------------------------------------------------------

    if model is None:

        print(
            "[CHEM][zero_shot] "
            "No pre-trained model supplied; attempting to train..."
        )

        model = _train_model_excluding(
            train_categories=train_categories,
            train_fn=train_fn,
            epochs=epochs,
        )

        result["trained"] = model is not None

    else:

        print(
            "[CHEM][zero_shot] "
            "Using supplied pre-trained model."
        )

    # ---------------------------------------------------------------
    # Step 2 — obtain test loader
    # ---------------------------------------------------------------

    if test_loader is None:

        test_loader = _build_held_out_loader(
            held_out_category=held_out_category
        )

    # ---------------------------------------------------------------
    # Step 3 — validate inputs
    # ---------------------------------------------------------------

    if model is None or test_loader is None:

        missing = []

        if model is None:
            missing.append("model")

        if test_loader is None:
            missing.append("test_loader")

        print(
            "[CHEM][zero_shot] Cannot evaluate. "
            f"Missing: {', '.join(missing)}"
        )

        return result

    # ---------------------------------------------------------------
    # Step 4 — evaluate
    # ---------------------------------------------------------------

    print(
        "[CHEM][zero_shot] "
        "Evaluating on held-out category..."
    )

    y_true, y_prob = _predict_probabilities(
        model,
        test_loader,
    )

    if y_true.size == 0:

        print(
            "[CHEM][zero_shot] "
            "Held-out loader produced no samples."
        )

        return result

    # ---------------------------------------------------------------
    # Step 5 — metrics
    # ---------------------------------------------------------------

    if threshold is None:
        try:
            threshold = get_coefficient(
                "final_harm_threshold"
            )
        except Exception:
            threshold = 0.5

    metrics = compute_metrics(
        y_true,
        y_prob,
        threshold=threshold,
    )

    result["metrics"] = metrics
    result["zero_shot_f1"] = metrics["f1"]
    result["evaluated"] = True

    print()
    print(
        f"[CHEM][zero_shot] "
        f"Results for unseen category "
        f"'{held_out_category}':"
    )

    pretty_print(metrics)

    print()
    print(
        f"[CHEM][zero_shot] "
        f">>> ZERO-SHOT F1 = {metrics['f1']:.4f} <<<"
    )

    return result


# ---------------------------------------------------------------------------
# Held-out loader
# ---------------------------------------------------------------------------

def _build_held_out_loader(
    held_out_category: str,
    data_dir: str = "data/hateful_memes",
    split: str = "dev",
) -> Optional[Any]:
    """
    Build a loader containing samples assigned to the held-out category.

    NOTE:
    Hateful Memes does not natively contain CHEM policy-category labels.
    Therefore this function only works when the preprocessing pipeline
    actually supplies a `category` field.
    """

    torch = _try_import_torch()

    if torch is None:
        return None

    try:
        from torch_geometric.loader import DataLoader
        from graph.graph_builder import build_graph
        from data.preprocess.hateful_memes import load_samples

    except Exception as exc:

        print(
            "[CHEM][zero_shot] "
            f"Data/graph pipeline unavailable: {exc}"
        )

        return None

    try:

        # IMPORTANT:
        # Explicitly specify the split and dataset directory.
        samples = load_samples(
            split,
            data_dir,
        )

    except TypeError:

        # Some versions may use keyword arguments.
        try:
            samples = load_samples(
                split=split,
                data_dir=data_dir,
            )

        except Exception as exc:

            print(
                "[CHEM][zero_shot] "
                f"Could not load samples: {exc}"
            )

            return None

    except Exception as exc:

        print(
            "[CHEM][zero_shot] "
            f"Could not load samples: {exc}"
        )

        return None

    graphs: List[Any] = []

    for sample in samples:

        category = sample.get("category")

        if category != held_out_category:
            continue

        elements = sample.get("elements")

        if not elements:
            continue

        try:

            built = build_graph(
                elements,
                label=int(sample.get("label", 1)),
            )

            graphs.append(
                built["data"]
            )

        except Exception as exc:

            print(
                f"[CHEM][zero_shot] "
                f"Skipping sample "
                f"{sample.get('id', '?')}: {exc}"
            )

    if not graphs:

        print(
            f"[CHEM][zero_shot] "
            f"No samples found for category "
            f"'{held_out_category}'."
        )

        return None

    print(
        f"[CHEM][zero_shot] "
        f"Built held-out loader with "
        f"{len(graphs)} graphs."
    )

    return DataLoader(
        graphs,
        batch_size=32,
        shuffle=False,
    )


# ---------------------------------------------------------------------------
# Checkpoint evaluation
# ---------------------------------------------------------------------------

def _run_with_checkpoint(
    held_out_category: str,
    data_dir: str,
    checkpoint: str,
    threshold: Optional[float],
) -> None:
    """
    Evaluate an existing GAT checkpoint on a held-out category.

    WARNING:
    The category assignment depends on the encoder's policy_category output.
    Hateful Memes itself does not provide CHEM categories.
    """

    import torch
    from torch_geometric.loader import DataLoader

    from model.gat_model import CHEMGraphAttentionNetwork
    from model.train import _load_samples

    from encoders.text_encoder import TextEncoder
    from encoders.image_encoder import ImageEncoder

    from PIL import Image

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ---------------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------------

    if not os.path.isfile(checkpoint):

        print(
            f"[CHEM][zero_shot] "
            f"Checkpoint does not exist: {checkpoint}"
        )

        return

    model = CHEMGraphAttentionNetwork().to(device)

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
    )

    state = checkpoint_data.get(
        "state_dict",
        checkpoint_data,
    )

    model.load_state_dict(state)

    model.eval()

    print(
        f"[CHEM][zero_shot] "
        f"Loaded checkpoint: {checkpoint}"
    )

    # ---------------------------------------------------------------
    # Encoders
    # ---------------------------------------------------------------

    text_enc = TextEncoder(
        device=str(device)
    )

    image_enc = ImageEncoder(
        device=str(device)
    )

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    print(
        "[CHEM][zero_shot] "
        "Loading test samples..."
    )

    raw_samples = _load_samples(
        "test",
        data_dir,
    )

    from graph.graph_builder import build_graph

    held_graphs: List[Any] = []

    for i, sample in enumerate(raw_samples):

        try:

            # -------------------------------------------------------
            # Text
            # -------------------------------------------------------

            text_out = text_enc.encode(
                sample.get("text", "")
            )

            elements = list(
                text_out.get("elements", [])
            )

            # -------------------------------------------------------
            # Image
            # -------------------------------------------------------

            img_field = sample.get("img")

            if img_field:

                img_path = (
                    img_field
                    if os.path.isabs(img_field)
                    else os.path.join(
                        data_dir,
                        img_field,
                    )
                )

                if os.path.isfile(img_path):

                    with Image.open(img_path) as image:

                        image_rgb = image.convert("RGB")

                        img_out = image_enc.encode(
                            image_rgb
                        )

                    elements.extend(
                        img_out.get("elements", [])
                    )

            # -------------------------------------------------------
            # Determine primary category
            # -------------------------------------------------------

            primary_category = None

            for element in elements:

                if not isinstance(element, dict):
                    continue

                if (
                    element.get("harmful_signal")
                    and element.get("policy_category")
                ):

                    primary_category = (
                        element["policy_category"]
                    )

                    break

            # -------------------------------------------------------
            # Only retain held-out category
            # -------------------------------------------------------

            if primary_category != held_out_category:
                continue

            label = int(
                sample.get("label", 0)
            )

            graph_dict = build_graph(
                elements,
                label,
            )

            held_graphs.append(
                graph_dict["data"]
            )

        except Exception as exc:

            print(
                f"[CHEM][zero_shot] "
                f"Skipping sample "
                f"{sample.get('id', i)}: {exc}"
            )

        if (i + 1) % 100 == 0:

            print(
                f"[CHEM][zero_shot] "
                f"Processed {i + 1}/{len(raw_samples)}"
            )

    # ---------------------------------------------------------------
    # Check result
    # ---------------------------------------------------------------

    print(
        f"[CHEM][zero_shot] "
        f"Held-out '{held_out_category}': "
        f"{len(held_graphs)} samples"
    )

    if not held_graphs:

        print(
            "[CHEM][zero_shot] "
            "No samples were assigned to the held-out category."
        )

        return

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    loader = DataLoader(
        held_graphs,
        batch_size=32,
        shuffle=False,
    )

    y_true: List[int] = []
    y_prob: List[float] = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            log_probs = model(
                batch.x.float(),
                batch.edge_index,
                batch.batch,
            )

            probs = torch.exp(
                log_probs
            )[:, 1]

            y_prob.extend(
                probs.cpu().tolist()
            )

            y_true.extend(
                batch.y.cpu().tolist()
            )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    if threshold is None:

        try:
            threshold = get_coefficient(
                "final_harm_threshold"
            )
        except Exception:
            threshold = 0.5

    metrics = compute_metrics(
        np.asarray(y_true),
        np.asarray(y_prob),
        threshold=threshold,
    )

    print()
    print(
        "[CHEM][zero_shot] "
        "=== ZERO-SHOT RESULTS ==="
    )

    pretty_print(metrics)

    print()

    print(
        f"[CHEM][zero_shot] "
        f">>> ZERO-SHOT F1 = "
        f"{metrics['f1']:.4f} <<<"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""

    default_categories: List[str] = []

    try:
        default_categories = list(
            get_harm_categories().keys()
        )
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description=(
            "CHEM zero-shot "
            "(leave-one-category-out) "
            "generalisation study."
        )
    )

    parser.add_argument(
        "--held-out-category",
        required=True,
        help=(
            "Harm category to hold out. "
            f"Choices: {default_categories}"
        ),
    )

    parser.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        type=str,
        default="data/hateful_memes",
        help="Hateful Memes dataset root.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/checkpoints/best_gat.pt",
        help="Path to trained GAT checkpoint.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Checkpoint mode
    # ---------------------------------------------------------------

    if os.path.isfile(args.checkpoint):

        print(
            f"[CHEM][zero_shot] "
            f"Found checkpoint: {args.checkpoint}"
        )

        _run_with_checkpoint(
            held_out_category=args.held_out_category,
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            threshold=args.threshold,
        )

    else:

        print(
            f"[CHEM][zero_shot] "
            f"No checkpoint found at: "
            f"{args.checkpoint}"
        )

        print(
            "[CHEM][zero_shot] "
            "Running train + evaluate mode."
        )

        run_zero_shot_eval(
            held_out_category=args.held_out_category,
            epochs=args.epochs,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()