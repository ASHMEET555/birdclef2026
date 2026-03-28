"""Feature engineering helpers for BirdCLEF 2026 precompute pipeline."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder


def make_label_vector(
    primary_label: str,
    secondary_labels: str,
    le: LabelEncoder,
    n_classes: int = 234,
    secondary_soft: float = 0.3,
    inat_to_primary: dict[str, str] | None = None,
) -> np.ndarray:
    """Create multi-label target vector with hard primary and soft secondary labels."""
    out = np.zeros(n_classes, dtype=np.float32)

    primary = str(primary_label) if primary_label is not None else ""
    if inat_to_primary and primary in inat_to_primary:
        primary = inat_to_primary[primary]

    if primary in le.classes_:
        out[int(le.transform([primary])[0])] = 1.0

    if isinstance(secondary_labels, str) and secondary_labels.strip() and secondary_labels != "[]":
        tokens = _parse_secondary_labels(secondary_labels)
        for token in tokens:
            if inat_to_primary and token in inat_to_primary:
                token = inat_to_primary[token]
            if token in le.classes_:
                idx = int(le.transform([token])[0])
                out[idx] = max(out[idx], np.float32(secondary_soft))

    return out


def make_soundscape_label(
    species_str: str,
    le: LabelEncoder,
    n_classes: int = 234,
    inat_to_primary: dict[str, str] | None = None,
) -> np.ndarray:
    """Build one multi-hot vector from semicolon-separated soundscape species labels."""
    out = np.zeros(n_classes, dtype=np.float32)
    if not isinstance(species_str, str) or not species_str.strip():
        return out

    for sp in [x.strip() for x in species_str.split(";") if x.strip()]:
        if inat_to_primary and sp in inat_to_primary:
            sp = inat_to_primary[sp]
        if sp in le.classes_:
            out[int(le.transform([sp])[0])] = 1.0
    return out


def build_inat_mapping(taxonomy_df) -> dict[str, str]:
    """Build mapping of iNat taxon IDs to primary_label strings."""
    mapping: dict[str, str] = {}
    for _, row in taxonomy_df.iterrows():
        inat_id = str(row.get("inat_taxon_id", "")).strip()
        primary = str(row.get("primary_label", "")).strip()
        if inat_id and primary:
            mapping[inat_id] = primary
    return mapping


def _parse_secondary_labels(secondary_labels: str) -> list[str]:
    """Parse secondary_labels string safely into a list of tokens."""
    try:
        parsed = ast.literal_eval(secondary_labels)
        if isinstance(parsed, list):
            return [str(x).strip().strip("'\"") for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass

    # Fallback: split on whitespace or semicolons
    cleaned = secondary_labels.replace(";", " ")
    return [tok.strip().strip("'\"") for tok in cleaned.split() if tok.strip()]


def compute_sample_weight(
    rating: float,
    is_inat: bool,
    species_count: int,
    total_count: int,
    gamma: float = -0.5,
    rarity_clip: float = 10.0,
) -> float:
    """Compute quality x rarity sample weight."""
    if is_inat:
        quality_weight = 0.5
    else:
        if rating is None or (isinstance(rating, float) and np.isnan(rating)) or float(rating) == 0.0:
            quality_weight = 0.5
        elif float(rating) >= 4.0:
            quality_weight = 1.0
        elif float(rating) >= 3.0:
            quality_weight = 0.8
        else:
            quality_weight = 0.6

    frac = max(float(species_count) / float(max(total_count, 1)), 1e-12)
    rarity_weight = float(math.pow(frac, gamma))
    rarity_weight = float(np.clip(rarity_weight, 0.0, rarity_clip))
    return float(quality_weight * rarity_weight)


def parse_soundscape_hour(filename: str) -> tuple[int, int]:
    """Parse hour/minute from BC2026 soundscape filename."""
    stem = Path(filename).stem
    hhmmss = stem.split("_")[-1]
    if len(hhmmss) != 6 or not hhmmss.isdigit():
        raise ValueError(f"Cannot parse HHMMSS from filename: {filename}")
    hour = int(hhmmss[:2])
    minute = int(hhmmss[2:4])
    return hour, minute


def cyclic_encode(value: float, period: float) -> tuple[float, float]:
    """Return sine/cosine cyclic encoding for periodic values."""
    angle = 2.0 * math.pi * (float(value) / float(period))
    return float(math.sin(angle)), float(math.cos(angle))
