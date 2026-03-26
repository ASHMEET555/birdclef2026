"""Evaluation metrics for BirdCLEF 2026.

Competition metric: Macro ROC-AUC across all 234 species.
Every species is weighted equally regardless of training sample count.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_macro_auc(
    probs: np.ndarray,
    labels: np.ndarray,
    species_list: Optional[list] = None,
) -> float:
    """Compute macro ROC-AUC across all species.

    WHY: Competition metric. Each species weighted equally (1/234), not weighted
    by sample count. A rare frog with 10 samples counts as much as a common bird
    with 1000 samples.

    Args:
        probs: Predicted probabilities [N, num_classes]
        labels: True binary labels [N, num_classes]
        species_list: Optional list of species names for filtering

    Returns:
        Macro-averaged ROC-AUC score (float)

    Notes:
        - Skips species with no positive samples (AUC undefined)
        - Skips species with all positive samples (AUC undefined)
        - Only computes AUC for species with at least one positive and one negative
    """
    if probs.shape != labels.shape:
        raise ValueError(f"Shape mismatch: probs {probs.shape} vs labels {labels.shape}")

    num_classes = probs.shape[1]
    valid_aucs = []

    for cls_idx in range(num_classes):
        cls_labels = labels[:, cls_idx]
        cls_probs = probs[:, cls_idx]

        # Skip if no variation in labels (all 0 or all 1)
        if len(np.unique(cls_labels)) < 2:
            continue

        try:
            auc = roc_auc_score(cls_labels, cls_probs)
            valid_aucs.append(auc)
        except ValueError:
            # Catch any edge cases (e.g., NaN in predictions)
            continue

    if len(valid_aucs) == 0:
        return 0.0

    # Macro average: equal weight to each class
    return float(np.mean(valid_aucs))


def compute_per_class_auc(
    probs: np.ndarray,
    labels: np.ndarray,
    taxonomy_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute AUC broken down by taxonomic class.

    WHY: Essential for paper ablation tables. Shows whether model is:
    - Solving birds well but failing on frogs (Amphibia)
    - Failing on zero-shot insects (Insecta)
    - Confusing mammal vocalizations (Mammalia)

    Args:
        probs: Predicted probabilities [N, num_classes]
        labels: True binary labels [N, num_classes]
        taxonomy_df: DataFrame with columns ['primary_label', 'class_name']
                     Maps species index to taxonomic class

    Returns:
        Dictionary with keys:
            'overall': macro AUC across all species
            'Aves': AUC for birds (162 species)
            'Amphibia': AUC for amphibians (32 species, primarily frogs)
            'Insecta': AUC for insects (28 species, 25 sonotypes)
            'Mammalia': AUC for mammals (8 species)
            'Reptilia': AUC for reptiles (1 species)

    Notes:
        - Taxonomy classes must match competition taxonomy.csv
        - Each class gets equal-weighted macro AUC within that class
        - Zero-shot species (no train_audio) will have low performance
    """
    if probs.shape != labels.shape:
        raise ValueError(f"Shape mismatch: probs {probs.shape} vs labels {labels.shape}")

    num_classes = probs.shape[1]

    # Build mapping from class index to taxonomic class
    # Assume taxonomy_df is ordered by label index 0-233
    if len(taxonomy_df) != num_classes:
        raise ValueError(f"Taxonomy df has {len(taxonomy_df)} rows but expected {num_classes}")

    class_to_indices = {}
    for idx, row in taxonomy_df.iterrows():
        class_name = row.get("class_name", row.get("class", "Unknown"))
        if class_name not in class_to_indices:
            class_to_indices[class_name] = []
        class_to_indices[class_name].append(idx)

    # Compute overall macro AUC
    overall_auc = compute_macro_auc(probs, labels)

    # Compute per-class AUC
    results = {"overall": overall_auc}

    for class_name in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        if class_name not in class_to_indices:
            results[class_name] = 0.0
            continue

        indices = class_to_indices[class_name]
        class_probs = probs[:, indices]
        class_labels = labels[:, indices]

        # Compute macro AUC within this taxonomic class
        class_auc = compute_macro_auc(class_probs, class_labels)
        results[class_name] = class_auc

    return results


def topn_postprocessing(
    preds: Dict[str, np.ndarray],
    n: int = 1,
) -> Dict[str, np.ndarray]:
    """TopN postprocessing: multiply predictions by file-level top-N mean.

    WHY: If a species is active in a file, it appears in multiple consecutive
    5-second windows. Multiplying each window's prediction by the file-level
    maximum increases signal for persistent species and suppresses transient noise.

    CONFIRMED: +0.011 LB improvement (2025 2nd place, TopN=1)

    Algorithm:
        For each file 'a' and each species 'c':
            top_n_mean[a, c] = mean of top-N predictions for species c in file a
            For each window 's' in file a:
                postproc_prob[a, s, c] = prob[a, s, c] × top_n_mean[a, c]

    Args:
        preds: Dict mapping filename → np.ndarray [num_windows, num_classes]
               Each file has multiple 5s windows
        n: Number of top predictions to average (default 1 = max)

    Returns:
        Dict with same structure but postprocessed probabilities

    Example:
        If file X has windows with frog predictions [0.1, 0.8, 0.7, 0.2]:
        - TopN=1 uses max=0.8 as multiplier
        - All windows get boosted: [0.08, 0.64, 0.56, 0.16]
        - Persistent frog signal amplified, transient noise suppressed
    """
    if n < 1:
        raise ValueError(f"TopN n must be >= 1, got {n}")

    postproc_preds = {}

    for filename, file_preds in preds.items():
        # file_preds: [num_windows, num_classes]
        num_windows, num_classes = file_preds.shape

        if num_windows < n:
            # If fewer windows than n, just use mean of all windows
            top_n_mean = file_preds.mean(axis=0)  # [num_classes]
        else:
            # For each species, take mean of top-N predictions
            # argsort returns indices in ascending order, so take last n
            top_n_mean = np.zeros(num_classes, dtype=np.float32)
            for cls_idx in range(num_classes):
                cls_preds = file_preds[:, cls_idx]
                # Get indices of top-n values
                top_n_indices = np.argsort(cls_preds)[-n:]
                top_n_mean[cls_idx] = cls_preds[top_n_indices].mean()

        # Multiply each window by the file-level top-N mean
        # Broadcasting: [num_windows, num_classes] * [num_classes]
        postproc_file_preds = file_preds * top_n_mean[np.newaxis, :]

        postproc_preds[filename] = postproc_file_preds

    return postproc_preds


def aggregate_predictions(
    window_preds: Dict[str, np.ndarray],
    aggregation: str = "max",
) -> Dict[str, np.ndarray]:
    """Aggregate window-level predictions to file-level predictions.

    Used for final submission where one probability per species per file is needed.

    Args:
        window_preds: Dict mapping filename → [num_windows, num_classes]
        aggregation: Method to aggregate ('max', 'mean', 'median')

    Returns:
        Dict mapping filename → [num_classes] single prediction per file

    Notes:
        - 'max' is standard for sparse presence detection
        - 'mean' can reduce false positives but may miss rare detections
        - 'median' is robust but may be too conservative
    """
    if aggregation not in ["max", "mean", "median"]:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    file_preds = {}

    for filename, preds in window_preds.items():
        # preds: [num_windows, num_classes]
        if aggregation == "max":
            file_pred = preds.max(axis=0)
        elif aggregation == "mean":
            file_pred = preds.mean(axis=0)
        elif aggregation == "median":
            file_pred = np.median(preds, axis=0)

        file_preds[filename] = file_pred

    return file_preds


# Smoke test
if __name__ == "__main__":
    print("Testing metrics...")

    # Create dummy data
    num_samples, num_classes = 100, 234
    np.random.seed(42)

    # Simulate predictions and labels
    probs = np.random.rand(num_samples, num_classes).astype(np.float32)
    labels = np.zeros((num_samples, num_classes), dtype=np.float32)

    # Add some positive labels (sparse, as in real competition)
    for i in range(num_samples):
        num_positives = np.random.randint(1, 5)  # 1-4 species per window
        positive_indices = np.random.choice(num_classes, size=num_positives, replace=False)
        labels[i, positive_indices] = 1.0

    # Test macro AUC
    macro_auc = compute_macro_auc(probs, labels)
    print(f"Macro AUC: {macro_auc:.4f}")
    assert 0.0 <= macro_auc <= 1.0, "AUC must be in [0, 1]"

    # Test per-class AUC
    # Create dummy taxonomy
    taxonomy_df = pd.DataFrame({
        "primary_label": [f"species_{i}" for i in range(num_classes)],
        "class_name": (
            ["Aves"] * 162 +
            ["Amphibia"] * 32 +
            ["Insecta"] * 28 +
            ["Mammalia"] * 8 +
            ["Reptilia"] * 4  # Padding to reach 234
        )[:num_classes]
    })

    per_class_auc = compute_per_class_auc(probs, labels, taxonomy_df)
    print(f"Per-class AUC:")
    for class_name, auc in per_class_auc.items():
        print(f"  {class_name}: {auc:.4f}")
        assert 0.0 <= auc <= 1.0, f"AUC for {class_name} must be in [0, 1]"

    # Test TopN postprocessing
    # Simulate file with multiple windows
    file_preds = {
        "file1.ogg": np.random.rand(10, num_classes).astype(np.float32),
        "file2.ogg": np.random.rand(15, num_classes).astype(np.float32),
        "file3.ogg": np.random.rand(5, num_classes).astype(np.float32),
    }

    postproc_preds = topn_postprocessing(file_preds, n=1)
    print(f"TopN postprocessing: processed {len(postproc_preds)} files")
    for filename in file_preds.keys():
        assert filename in postproc_preds, f"Missing {filename} in postproc"
        assert postproc_preds[filename].shape == file_preds[filename].shape

    # Test aggregation
    agg_preds = aggregate_predictions(file_preds, aggregation="max")
    print(f"Aggregation: {len(agg_preds)} files")
    for filename, pred in agg_preds.items():
        assert pred.shape == (num_classes,), f"Expected shape ({num_classes},), got {pred.shape}"
        assert 0.0 <= pred.min() and pred.max() <= 1.0, "Predictions out of range"

    print("✓ All metrics work correctly")
    print("OK")
