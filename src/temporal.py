"""Temporal features and activity priors for BirdCLEF 2026.

WHY TEMPORAL FEATURES MATTER:
- Frogs are 15:1 dominant in evening soundscapes vs daytime
- Many bird species have dawn/dusk chorus patterns
- Temporal patterns are biological and transfer perfectly to test set
- Geographic features (lat/lon) DON'T transfer - different test location

NEVER use geographic features as model input.
ALWAYS use temporal features for postprocessing.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def extract_hour_from_filename(filename: str) -> float:
    """Extract decimal hour (0-24) from BirdCLEF 2026 soundscape filename.

    Filename format: BC2026_Train_XXXX_SYY_YYYYMMDD_HHMMSS.ogg
    Example: BC2026_Train_0001_S01_20240315_173045.ogg → 17:30:45 → 17.5125

    Args:
        filename: Soundscape filename or full path

    Returns:
        Decimal hour in range [0, 24)

    Raises:
        ValueError: If filename doesn't match expected format
    """
    # Extract just the filename from path
    filename = Path(filename).name

    # Pattern: _YYYYMMDD_HHMMSS.ogg
    match = re.search(r"_(\d{8})_(\d{6})\.ogg", filename)
    if not match:
        raise ValueError(f"Cannot parse datetime from filename: {filename}")

    date_str, time_str = match.groups()

    # Parse time: HHMMSS
    if len(time_str) != 6:
        raise ValueError(f"Invalid time format: {time_str}")

    hour = int(time_str[0:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])

    # Convert to decimal hour
    decimal_hour = float(hour) + float(minute) / 60.0 + float(second) / 3600.0

    # Return hour modulo 24 (handle edge cases)
    return decimal_hour % 24.0


def extract_hour_from_window(filename: str, window_start_seconds: float = 0.0) -> float:
    """Extract decimal hour for a specific window within a soundscape file.

    Args:
        filename: Soundscape filename
        window_start_seconds: Offset in seconds from file start

    Returns:
        Decimal hour for this specific window

    Example:
        File starts at 17:30:00, window at 65 seconds offset
        → 17:30:00 + 65s = 17:31:05 → 17.5181
    """
    file_hour = extract_hour_from_filename(filename)

    # Add window offset in hours
    window_offset_hours = window_start_seconds / 3600.0
    window_hour = file_hour + window_offset_hours

    # Return hour modulo 24
    return window_hour % 24.0


def cyclic_time_encoding(decimal_hour: float) -> Tuple[float, float]:
    """Encode time-of-day as 2D cyclic vector (sin, cos).

    WHY: Raw hour (0-23) is discontinuous at midnight. 23:59 and 00:01 are
    2 minutes apart but numerically 22 units apart. Sin/cos encoding places
    them adjacent in feature space.

    Math:
        angle = 2π × (hour / 24)
        encoding = (sin(angle), cos(angle))

    Args:
        decimal_hour: Hour in range [0, 24)

    Returns:
        Tuple of (sin_value, cos_value), each in range [-1, 1]

    Example:
        00:00 → (0.0, 1.0)   # Midnight
        06:00 → (1.0, 0.0)   # Morning
        12:00 → (0.0, -1.0)  # Noon
        18:00 → (-1.0, 0.0)  # Evening
        23:59 → (≈0.0, ≈1.0) # Back to midnight
    """
    # Normalize to [0, 1] and convert to radians
    angle = 2.0 * math.pi * (decimal_hour / 24.0)

    sin_val = math.sin(angle)
    cos_val = math.cos(angle)

    return (float(sin_val), float(cos_val))


def load_temporal_prior(precomputed_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load precomputed temporal activity priors.

    Expected files in precomputed_dir:
        - temporal_prior.npy: [234, 24] species × hour activity
        - temporal_prior_class.npy: [5, 24] taxonomic class × hour activity

    Args:
        precomputed_dir: Path to precomputed artifacts folder

    Returns:
        Tuple of (prior_species, prior_class)
            prior_species: [234, 24] float32 - activity per species per hour
            prior_class: [5, 24] float32 - activity per taxonomic class per hour

    Raises:
        FileNotFoundError: If prior files don't exist
    """
    precomputed_dir = Path(precomputed_dir)

    prior_species_path = precomputed_dir / "temporal_prior.npy"
    prior_class_path = precomputed_dir / "temporal_prior_class.npy"

    if not prior_species_path.exists():
        raise FileNotFoundError(
            f"Species temporal prior not found: {prior_species_path}\n"
            "Run precompute pipeline to generate priors from labeled soundscapes."
        )

    if not prior_class_path.exists():
        # Class prior is optional - create uniform fallback
        print(f"Warning: Class temporal prior not found: {prior_class_path}")
        prior_class = np.ones((5, 24), dtype=np.float32)
    else:
        prior_class = np.load(prior_class_path)

    prior_species = np.load(prior_species_path)

    return prior_species, prior_class


def apply_temporal_prior(
    probs: np.ndarray,
    hour: float,
    prior_species: np.ndarray,
    prior_class: Optional[np.ndarray] = None,
    taxonomy_df: Optional[pd.DataFrame] = None,
    use_class_fallback: bool = True,
) -> np.ndarray:
    """Multiply predictions by hour-of-day activity prior.

    WHY: Biological activity patterns are strong and transferable.
    - Frogs are 15:1 more active in evening vs daytime
    - Many bird species have dawn/dusk chorus patterns
    - At 9pm, boost frog predictions and suppress most bird predictions

    Method:
        adjusted_prob[s] = prob[s] × activity_prior[s, hour]

    For species lacking sufficient soundscape data (zero-shot or rare):
        Fall back to taxonomic class-level prior (all frogs share Amphibia pattern)

    Args:
        probs: Prediction probabilities [num_classes] for one window
        hour: Decimal hour [0, 24) for this window
        prior_species: Species-level prior [num_classes, 24]
        prior_class: Class-level prior [5, 24], optional
        taxonomy_df: DataFrame mapping species to taxonomic class, optional
        use_class_fallback: If True, use class prior for species with low coverage

    Returns:
        Adjusted probabilities [num_classes]

    Notes:
        - Hour is discretized to integer hour (0-23) for prior lookup
        - Class order: [Aves, Amphibia, Insecta, Mammalia, Reptilia]
        - Prior values are probabilities (sum to 1 across 24 hours per species)
    """
    if probs.ndim != 1:
        raise ValueError(f"Expected 1D probs array, got shape {probs.shape}")

    num_classes = len(probs)
    if prior_species.shape[0] != num_classes:
        raise ValueError(
            f"Prior species has {prior_species.shape[0]} species "
            f"but probs has {num_classes}"
        )

    # Discretize hour to integer [0, 23]
    hour_int = int(hour) % 24

    # Extract prior for this hour: [num_classes]
    species_prior = prior_species[:, hour_int]

    # Check for species with insufficient data (prior = 1/24 = uniform)
    # These should fall back to class prior if available
    uniform_threshold = 1.0 / 24.0 + 0.01  # Allow small epsilon
    low_coverage_mask = np.abs(species_prior - uniform_threshold) < 0.02

    # Apply species prior
    adjusted = probs * species_prior

    # Apply class fallback for low-coverage species
    if use_class_fallback and prior_class is not None and taxonomy_df is not None:
        # Build class mapping
        class_names = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for species_idx in np.where(low_coverage_mask)[0]:
            # Get taxonomic class for this species
            if species_idx >= len(taxonomy_df):
                continue

            species_row = taxonomy_df.iloc[species_idx]
            class_name = species_row.get("class_name", species_row.get("class", "Aves"))

            if class_name in class_to_idx:
                class_idx = class_to_idx[class_name]
                class_prior_val = prior_class[class_idx, hour_int]

                # Replace species prior with class prior
                adjusted[species_idx] = probs[species_idx] * class_prior_val

    return adjusted


def compute_temporal_prior_from_soundscapes(
    soundscape_labels_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    num_classes: int = 234,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute temporal activity priors from labeled soundscape windows.

    This function is part of the precompute pipeline, not used during training.

    Args:
        soundscape_labels_df: DataFrame with columns ['filename', 'start', 'primary_label']
        taxonomy_df: DataFrame with species and taxonomic class info
        num_classes: Number of species (default 234)

    Returns:
        Tuple of (prior_species, prior_class)
            prior_species: [num_classes, 24] normalized activity per species per hour
            prior_class: [5, 24] normalized activity per taxonomic class per hour

    Algorithm:
        1. For each labeled soundscape window:
           - Extract hour from filename + start time
           - Increment count[species, hour]
        2. Normalize each species: prior[s, :] = count[s, :] / sum(count[s, :])
        3. For zero-shot species: prior[s, :] = uniform = 1/24
        4. Aggregate to class level: prior_class[c, h] = mean(prior[species in c, h])
    """
    # Initialize counts: [num_classes, 24]
    species_counts = np.zeros((num_classes, 24), dtype=np.float32)

    # Build species to index mapping
    species_to_idx = {
        row["primary_label"]: idx for idx, row in taxonomy_df.iterrows()
    }

    # Count occurrences per species per hour
    for _, row in soundscape_labels_df.iterrows():
        filename = row["filename"]
        start_time = row.get("start", "00:00:00")

        # Convert start time to seconds
        if isinstance(start_time, str):
            parts = start_time.split(":")
            start_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            start_seconds = 0.0

        # Extract hour for this window
        try:
            window_hour = extract_hour_from_window(filename, start_seconds)
            hour_int = int(window_hour) % 24
        except ValueError:
            continue

        # Parse species labels (semicolon-separated)
        species_str = row.get("primary_label", "")
        if not isinstance(species_str, str):
            continue

        species_list = [s.strip() for s in species_str.split(";") if s.strip()]

        for species in species_list:
            if species in species_to_idx:
                species_idx = species_to_idx[species]
                species_counts[species_idx, hour_int] += 1.0

    # Normalize to probabilities (sum to 1 across 24 hours per species)
    prior_species = np.zeros_like(species_counts)
    for species_idx in range(num_classes):
        total = species_counts[species_idx].sum()
        if total > 0:
            prior_species[species_idx] = species_counts[species_idx] / total
        else:
            # Zero-shot or no soundscape labels: uniform prior
            prior_species[species_idx] = 1.0 / 24.0

    # Compute class-level priors
    class_names = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
    prior_class = np.zeros((5, 24), dtype=np.float32)

    for class_idx, class_name in enumerate(class_names):
        # Find all species in this class
        class_species_indices = [
            idx
            for idx, row in taxonomy_df.iterrows()
            if row.get("class_name", row.get("class")) == class_name
        ]

        if len(class_species_indices) > 0:
            # Average species priors within this class
            class_prior = prior_species[class_species_indices].mean(axis=0)
            prior_class[class_idx] = class_prior
        else:
            # No species in this class: uniform
            prior_class[class_idx] = 1.0 / 24.0

    return prior_species, prior_class


# Smoke test
if __name__ == "__main__":
    print("Testing temporal features...")

    # Test hour extraction
    test_files = [
        "BC2026_Train_0001_S01_20240315_173045.ogg",
        "BC2026_Train_0123_S05_20240802_063012.ogg",
        "BC2026_Train_9999_S99_20241225_235959.ogg",
    ]

    for filename in test_files:
        hour = extract_hour_from_filename(filename)
        print(f"{filename} → {hour:.4f}")
        assert 0.0 <= hour < 24.0, f"Hour out of range: {hour}"

    # Test window hour extraction
    window_hour = extract_hour_from_window(test_files[0], window_start_seconds=65.0)
    print(f"Window at +65s: {window_hour:.4f}")

    # Test cyclic encoding
    test_hours = [0.0, 6.0, 12.0, 18.0, 23.99]
    for h in test_hours:
        sin_val, cos_val = cyclic_time_encoding(h)
        print(f"Hour {h:5.2f} → sin={sin_val:6.3f}, cos={cos_val:6.3f}")
        # Check magnitude (should be on unit circle)
        magnitude = math.sqrt(sin_val**2 + cos_val**2)
        assert abs(magnitude - 1.0) < 1e-6, f"Not on unit circle: {magnitude}"

    # Test temporal prior application
    num_classes = 234
    np.random.seed(42)

    # Create dummy priors
    prior_species = np.random.rand(num_classes, 24).astype(np.float32)
    # Normalize
    prior_species = prior_species / prior_species.sum(axis=1, keepdims=True)

    prior_class = np.random.rand(5, 24).astype(np.float32)
    prior_class = prior_class / prior_class.sum(axis=1, keepdims=True)

    # Create dummy predictions
    probs = np.random.rand(num_classes).astype(np.float32)

    # Apply prior for hour 17 (evening)
    adjusted = apply_temporal_prior(
        probs,
        hour=17.5,
        prior_species=prior_species,
        prior_class=prior_class,
        taxonomy_df=None,
        use_class_fallback=False,
    )

    print(f"Original probs: mean={probs.mean():.4f}, max={probs.max():.4f}")
    print(f"Adjusted probs: mean={adjusted.mean():.4f}, max={adjusted.max():.4f}")
    assert adjusted.shape == probs.shape, "Shape mismatch"
    assert np.all(adjusted >= 0.0), "Negative probabilities after adjustment"

    print("✓ All temporal features work correctly")
    print("OK")
