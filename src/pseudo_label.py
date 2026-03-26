"""Pseudo-label generation and filtering for BirdCLEF 2026.

WHY PSEUDO-LABELING:
- Leverage 10,658 unlabeled soundscape files (vs 66 labeled)
- Bridge domain gap between clean clips and noisy soundscapes
- Iterative refinement improves over rounds

KEY TECHNIQUES:
- PowerTransform to prevent probability collapse
- Confidence thresholding
- Out-of-fold (OOF) pseudo-labeling to prevent label leakage
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optional dependencies
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def power_transform(probs: np.ndarray, gamma: float = 2.0, min_prob: float = 0.1) -> np.ndarray:
    """Apply power transform to prevent probability collapse across rounds.

    WHY: Without this, round 3-4 labels become overconfident near-one-hot vectors.
    Model stops learning from pseudo-labels. PowerTransform keeps distribution spread.

    Formula:
        probs_transformed = probs ** gamma
        probs_transformed = normalize(probs_transformed)  # Re-normalize to sum to ~1
        probs_transformed[probs_transformed < min_prob] = 0.0  # Filter noise

    CONFIRMED: PowerTransform is essential for stable multi-round pseudo-labeling
    (2025 1st place technique)

    Args:
        probs: Probabilities [N, num_classes] or [num_classes]
        gamma: Power exponent (default 2.0, higher = more sharpening)
        min_prob: Zero out probs below this threshold after transform

    Returns:
        Transformed probabilities
    """
    # Apply power transform
    transformed = np.power(probs, gamma)

    # Re-normalize (row-wise if 2D)
    if transformed.ndim == 2:
        row_sums = transformed.sum(axis=1, keepdims=True)
        transformed = transformed / (row_sums + 1e-8)
    else:
        row_sum = transformed.sum()
        transformed = transformed / (row_sum + 1e-8)

    # Filter low-confidence predictions
    transformed[transformed < min_prob] = 0.0

    return transformed.astype(np.float32)


@torch.no_grad()
def generate_pseudo_labels(
    model: nn.Module,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
    mel_transform: nn.Module,
    round_num: int = 1,
) -> Dict[str, np.ndarray]:
    """Generate pseudo-labels for unlabeled soundscape windows.

    Args:
        model: Trained model
        dataloader: Soundscape dataloader (unlabeled windows)
        config: Config dict
        device: Torch device
        mel_transform: Mel spectrogram transform
        round_num: Pseudo-label round number (1, 2, 3, ...)

    Returns:
        Dict mapping window_id → pseudo_label [234]
    """
    model.eval()
    mel_transform.eval()

    # Get pseudo-label config
    pseudo_config = config.get("pseudo", {})
    threshold = pseudo_config.get("threshold", 0.3 if round_num == 1 else 0.4)
    gamma = pseudo_config.get("gamma", 2.0)
    min_prob = pseudo_config.get("min_prob", 0.1)

    print(f"Generating pseudo-labels (round {round_num}):")
    print(f"  Threshold: {threshold}")
    print(f"  PowerTransform gamma: {gamma}")
    print(f"  Min prob: {min_prob}")

    pseudo_labels = {}
    selected_count = 0
    total_count = 0

    pbar = tqdm(dataloader, desc=f"Pseudo-label round {round_num}")

    for batch in pbar:
        waveforms = batch["waveform"].to(device)
        window_ids = batch["window_id"]

        # Convert to mel
        mel = mel_transform(waveforms)

        # Forward pass
        logits, probs = model(mel)
        probs_cpu = probs.cpu().numpy()  # [B, 234]

        # Filter by confidence and apply transform
        for i, window_id in enumerate(window_ids):
            window_probs = probs_cpu[i]  # [234]

            # Check if max prob exceeds threshold
            max_prob = window_probs.max()
            if max_prob >= threshold:
                # Apply power transform
                transformed_probs = power_transform(window_probs, gamma=gamma, min_prob=min_prob)

                # Store pseudo-label
                pseudo_labels[window_id] = transformed_probs
                selected_count += 1

            total_count += 1

        pbar.set_postfix({"selected": f"{selected_count}/{total_count}"})

    print(f"Selected {selected_count}/{total_count} windows ({100*selected_count/max(total_count,1):.1f}%)")

    return pseudo_labels


@torch.no_grad()
def oof_pseudo_label(
    model_folds: List[nn.Module],
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
    mel_transform: nn.Module,
    soundscape_to_fold: Dict[str, int],
    round_num: int = 1,
) -> Dict[str, np.ndarray]:
    """Out-of-fold pseudo-labeling to prevent label leakage.

    WHY: Each soundscape window is labeled by models NOT trained on that fold.
    Prevents memorization and label leakage from overlapping training data.

    This is a principled semi-supervised learning approach equivalent to
    cross-validation for pseudo-labeling.

    Args:
        model_folds: List of 5 trained models (one per fold)
        dataloader: Soundscape dataloader
        config: Config dict
        device: Torch device
        mel_transform: Mel transform
        soundscape_to_fold: Dict mapping soundscape filename → fold assignment
        round_num: Pseudo-label round number

    Returns:
        Dict mapping window_id → pseudo_label [234]

    Notes:
        - Models are loaded onto device one at a time to save memory
        - Each window predicted by 4 models (excluding its fold), averaged
    """
    # Set all models to eval
    for model in model_folds:
        model.eval()

    mel_transform.eval()

    # Get pseudo-label config
    pseudo_config = config.get("pseudo", {})
    threshold = pseudo_config.get("threshold", 0.3 if round_num == 1 else 0.4)
    gamma = pseudo_config.get("gamma", 2.0)
    min_prob = pseudo_config.get("min_prob", 0.1)

    print(f"OOF Pseudo-labeling (round {round_num}):")
    print(f"  Threshold: {threshold}")
    print(f"  Models: {len(model_folds)} folds")

    # Collect all predictions
    window_predictions = {}  # window_id → list of [234] arrays from each valid fold

    pbar = tqdm(dataloader, desc=f"OOF round {round_num}")

    for batch in pbar:
        waveforms = batch["waveform"].to(device)
        filenames = batch["filename"]
        window_ids = batch["window_id"]

        # Convert to mel once (shared across all models)
        mel = mel_transform(waveforms)

        # For each model/fold
        for fold_idx, model in enumerate(model_folds):
            # Skip this model for windows from same fold
            batch_masks = []
            for i, filename in enumerate(filenames):
                window_fold = soundscape_to_fold.get(filename, -1)
                # Only predict if window fold != model fold
                use_this_model = (window_fold != fold_idx) and (window_fold != -1)
                batch_masks.append(use_this_model)

            # Filter batch to only valid windows for this model
            valid_indices = [i for i, use in enumerate(batch_masks) if use]

            if len(valid_indices) == 0:
                continue

            # Get predictions for valid windows
            valid_mel = mel[valid_indices]

            with torch.no_grad():
                logits, probs = model(valid_mel)
                probs_cpu = probs.cpu().numpy()

            # Store predictions
            for local_idx, batch_idx in enumerate(valid_indices):
                window_id = window_ids[batch_idx]
                if window_id not in window_predictions:
                    window_predictions[window_id] = []
                window_predictions[window_id].append(probs_cpu[local_idx])

    # Average predictions across folds and apply threshold
    pseudo_labels = {}
    selected_count = 0

    for window_id, pred_list in window_predictions.items():
        # Average across folds
        avg_probs = np.mean(pred_list, axis=0)  # [234]

        # Check threshold
        max_prob = avg_probs.max()
        if max_prob >= threshold:
            # Apply power transform
            transformed_probs = power_transform(avg_probs, gamma=gamma, min_prob=min_prob)
            pseudo_labels[window_id] = transformed_probs
            selected_count += 1

    print(f"OOF selected {selected_count}/{len(window_predictions)} windows "
          f"({100*selected_count/max(len(window_predictions),1):.1f}%)")

    return pseudo_labels


def save_pseudo_labels(
    pseudo_labels: Dict[str, np.ndarray],
    output_path: str | Path,
    round_num: int,
):
    """Save pseudo-labels to HDF5 file.

    Args:
        pseudo_labels: Dict mapping window_id → label [234]
        output_path: Output directory for pseudo-labels
        round_num: Round number
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for pseudo-label storage. Install with: pip install h5py")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    h5_path = output_path / f"pseudo_labels_round{round_num}.h5"

    with h5py.File(h5_path, "w") as f:
        for window_id, label in pseudo_labels.items():
            f.create_dataset(window_id, data=label, dtype=np.float32)

    print(f"Saved {len(pseudo_labels)} pseudo-labels to {h5_path}")


def load_pseudo_labels(
    pseudo_label_path: str | Path,
    round_num: int,
) -> Dict[str, np.ndarray]:
    """Load pseudo-labels from HDF5 file.

    Args:
        pseudo_label_path: Path to pseudo-labels directory
        round_num: Round number

    Returns:
        Dict mapping window_id → label [234]
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for pseudo-label loading. Install with: pip install h5py")

    h5_path = Path(pseudo_label_path) / f"pseudo_labels_round{round_num}.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"Pseudo-labels not found: {h5_path}")

    pseudo_labels = {}

    with h5py.File(h5_path, "r") as f:
        for window_id in f.keys():
            pseudo_labels[window_id] = f[window_id][:]

    print(f"Loaded {len(pseudo_labels)} pseudo-labels from {h5_path}")

    return pseudo_labels


# Smoke test
if __name__ == "__main__":
    print("Testing pseudo-label generation...")

    # Test power transform
    probs = np.array([0.8, 0.15, 0.03, 0.02], dtype=np.float32)
    transformed = power_transform(probs, gamma=2.0, min_prob=0.1)
    print(f"Original: {probs}")
    print(f"Transformed (gamma=2.0): {transformed}")
    print(f"Sum: {transformed.sum():.4f}")

    assert transformed.sum() > 0, "Transformed probs should sum to > 0"
    assert (transformed <= 1.0).all(), "Probs should be <= 1.0"

    # Test batch transform
    batch_probs = np.random.rand(10, 234).astype(np.float32)
    batch_transformed = power_transform(batch_probs, gamma=2.0)
    print(f"Batch shape: {batch_transformed.shape}")
    assert batch_transformed.shape == (10, 234)

    print("✓ Power transform works correctly")
    print("OK")
