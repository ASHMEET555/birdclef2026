"""Dataset definitions for BirdCLEF 2026.

Loads precomputed artifacts and provides clean data access for training.

Key principles:
- Load precomputed labels/weights ONCE in __init__, not per item
- Tile-repeat audio shorter than 5s (NO zero-padding)
- Return dictionaries with all metadata for flexible training loops
- Support Perch soft labels with graceful fallback to hard labels
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Optional dependencies - lazy import to avoid hard requirement
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# Known corrupted files - EXCLUDE from all datasets
CORRUPTED_FILES = {
    "209233/iNat1545859.ogg",
    "47144/iNat1191939.ogg",
    "47144/iNat1317365.ogg",
    "47144/iNat1317366.ogg",
}

# Known noisy species - DOWN-WEIGHT (do not delete)
DOWNWEIGHT_SPECIES = {
    "47144": 0.3,  # Domestic Dog - background noise mislabeled as species
}


class FocalDataset(Dataset):
    """Dataset for train_audio focal clips with precomputed labels and weights.

    WHY PRECOMPUTED: Speeds up training by pre-computing expensive operations:
    - Multi-label vectors with secondary species
    - Sample weights (1/sqrt(class_count) per clip)
    - CV fold assignments
    - Perch soft labels from teacher model

    Returns dict with keys:
        waveform: [160000] raw audio mono 32kHz
        mel: [1, 128, 313] mel spectrogram after PCEN (computed lazily)
        label: [234] final label vector (hard + optional Perch blending)
        sample_weight: float for WeightedRandomSampler
        filename: str for debugging
        class_name: str taxonomic class for per-class AUC
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        config: dict,
        precomputed_dir: str | Path,
        mode: str = "train",
        use_perch_labels: bool = False,
        perch_alpha: float = 0.7,
        transform: Optional[torch.nn.Module] = None,
    ):
        """Initialize FocalDataset.

        Args:
            meta_df: DataFrame from train_folds.csv (filtered by fold)
            config: Config dict with audio/paths settings
            precomputed_dir: Path to precomputed/ folder
            mode: 'train' or 'val' (affects random cropping)
            use_perch_labels: Whether to blend Perch soft labels
            perch_alpha: Blend weight for Perch (0=hard only, 1=Perch only)
            transform: Optional mel spectrogram transform
        """
        self.meta_df = meta_df.reset_index(drop=True)
        self.config = config
        self.precomputed_dir = Path(precomputed_dir)
        self.mode = mode
        self.use_perch_labels = use_perch_labels
        self.perch_alpha = perch_alpha
        self.transform = transform

        # Audio config
        self.sample_rate = config.get("audio", {}).get("sample_rate", 32000)
        self.chunk_duration = config.get("audio", {}).get("chunk_duration", 5.0)
        self.n_samples = int(self.sample_rate * self.chunk_duration)  # 160000

        self.audio_root = Path(config.get("paths", {}).get("train_audio", "./data/train_audio"))

        # Load precomputed labels [num_samples, 234]
        labels_path = self.precomputed_dir / "label_vectors" / "train_labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"Precomputed labels not found: {labels_path}")

        self.hard_labels = np.load(labels_path)  # [35186, 234]

        # Load sample weights [num_samples]
        weights_path = self.precomputed_dir / "metadata" / "sample_weights.npy"
        if not weights_path.exists():
            print(f"Warning: Sample weights not found: {weights_path}, using uniform weights")
            self.sample_weights = np.ones(len(self.hard_labels), dtype=np.float32)
        else:
            self.sample_weights = np.load(weights_path)

        # Load taxonomy for class mapping
        taxonomy_path = config.get("paths", {}).get("taxonomy", "./data/taxonomy.csv")
        self.taxonomy_df = pd.read_csv(taxonomy_path)

        # Build species ID to label index mapping
        self.species_to_idx = {
            str(row["primary_label"]): idx for idx, row in self.taxonomy_df.iterrows()
        }

        # Identify zero-shot species (no train_audio samples)
        self.zero_shot_species = set()
        species_counts = self.hard_labels.sum(axis=0)
        for idx, count in enumerate(species_counts):
            if count == 0:
                self.zero_shot_species.add(idx)

        # Optionally load Perch soft labels
        self.perch_labels = None
        if use_perch_labels:
            self.perch_labels = self._load_perch_labels()

        print(f"FocalDataset {mode}: {len(self.meta_df)} samples")
        print(f"  Hard labels shape: {self.hard_labels.shape}")
        print(f"  Zero-shot species: {len(self.zero_shot_species)}")
        if self.perch_labels is not None:
            print(f"  Perch labels: {len(self.perch_labels)} files")

    def _load_perch_labels(self) -> Optional[dict]:
        """Load Perch soft labels from individual .npy embedding files.

        Returns dict mapping filename → [234] soft label vector, or None if not available.
        """
        if not HAS_H5PY:
            print("h5py not available - using hard labels only")
            return None

        perch_dir = self.precomputed_dir / "perch_embeddings" / "train_audio"
        if not perch_dir.exists():
            print(f"Perch embeddings directory not found: {perch_dir}, using hard labels only")
            return None

        # Build mapping from filename → soft labels
        # NOTE: This assumes Perch embeddings are already converted to [234] label space
        # If they're still embeddings, you'll need to load a Perch→label projection matrix
        perch_labels = {}

        for idx, row in self.meta_df.iterrows():
            filename = row["filename"]
            species_id = Path(filename).parts[0]
            file_stem = Path(filename).stem

            perch_file = perch_dir / species_id / f"{file_stem}.npy"

            if perch_file.exists():
                try:
                    # Load Perch soft labels [234]
                    # NOTE: Adjust this if Perch files are embeddings not labels
                    perch_label = np.load(perch_file, allow_pickle=True)

                    # Handle different Perch output formats
                    if isinstance(perch_label, np.ndarray):
                        if perch_label.shape == (234,):
                            perch_labels[filename] = perch_label
                    elif isinstance(perch_label, dict):
                        # If Perch saves dict with 'logits' or 'probabilities' key
                        if "probabilities" in perch_label:
                            probs = perch_label["probabilities"]
                            if probs.shape == (234,):
                                perch_labels[filename] = probs
                except Exception as e:
                    # Silently skip files that fail to load
                    pass

        if len(perch_labels) == 0:
            print("Warning: No valid Perch labels found, falling back to hard labels only")
            return None

        return perch_labels

    def __len__(self) -> int:
        return len(self.meta_df)

    def _load_audio(self, filename: str) -> torch.Tensor:
        """Load audio file and prepare 5-second chunk.

        Args:
            filename: Relative path like "species_id/filename.ogg"

        Returns:
            Waveform tensor [160000] mono 32kHz
        """
        if not HAS_LIBROSA:
            raise ImportError("librosa is required for audio loading. Install with: pip install librosa")

        audio_path = self.audio_root / filename

        # Load with librosa (always mono, always 32kHz)
        try:
            waveform, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
            )
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return silence as fallback
            return torch.zeros(self.n_samples, dtype=torch.float32)

        # Handle duration
        length = len(waveform)

        if length < self.n_samples:
            # TILE-REPEAT to fill 160000 samples (NO zero-padding)
            # WHY: Padding with zeros creates artificial silence that doesn't exist
            # in real data. Tiling repeats the actual species call.
            n_repeats = (self.n_samples // length) + 1
            waveform = np.tile(waveform, n_repeats)[:self.n_samples]

        elif length > self.n_samples:
            if self.mode == "train":
                # Random crop for training (augmentation)
                max_offset = length - self.n_samples
                offset = np.random.randint(0, max_offset + 1)
            else:
                # Deterministic crop from start for validation
                offset = 0

            waveform = waveform[offset : offset + self.n_samples]

        else:
            # Exactly 5 seconds - use as-is
            pass

        return torch.from_numpy(waveform).float()

    def __getitem__(self, idx: int) -> dict:
        """Get one sample.

        Returns dict with keys:
            waveform: [160000]
            label: [234]
            sample_weight: float
            filename: str
            class_name: str
            primary_label: str
        """
        row = self.meta_df.iloc[idx]
        filename = row["filename"]

        # Load audio
        waveform = self._load_audio(filename)

        # Get hard label for this sample (using meta_df index in original labels array)
        # Find the row index in the full train_folds.csv to match precomputed labels
        original_idx = row.name if "name" in row.index else idx
        hard_label = self.hard_labels[idx].copy()  # [234]

        # Blend with Perch soft labels if available
        if self.perch_labels is not None and filename in self.perch_labels:
            perch_label = self.perch_labels[filename]

            # Blend: alpha * Perch + (1-alpha) * hard
            # But keep hard labels for zero-shot species (Perch can't predict them)
            label = self.perch_alpha * perch_label + (1 - self.perch_alpha) * hard_label

            # Force zero-shot species to use hard labels
            for species_idx in self.zero_shot_species:
                label[species_idx] = hard_label[species_idx]

        else:
            label = hard_label

        # Apply species down-weighting (e.g., Domestic Dog noise)
        primary_label_id = str(row.get("primary_label", ""))
        if primary_label_id in DOWNWEIGHT_SPECIES:
            downweight_factor = DOWNWEIGHT_SPECIES[primary_label_id]
            # Find species index
            if primary_label_id in self.species_to_idx:
                species_idx = self.species_to_idx[primary_label_id]
                label[species_idx] *= downweight_factor

        # Get sample weight
        sample_weight = self.sample_weights[idx]

        # Get class name for per-class AUC tracking
        class_name = row.get("class_name", "Unknown")

        return {
            "waveform": waveform,
            "label": torch.from_numpy(label).float(),
            "sample_weight": float(sample_weight),
            "filename": filename,
            "class_name": class_name,
            "primary_label": primary_label_id,
        }


class SoundscapeDataset(Dataset):
    """Dataset for soundscape windows (labeled or unlabeled).

    Returns dict with keys:
        waveform: [160000]
        label: [234] (zeros if unlabeled)
        has_label: bool
        hour: float decimal hour
        time_sin: float sin(2π×hour/24)
        time_cos: float cos(2π×hour/24)
        filename: str
        window_id: str for submission
    """

    def __init__(
        self,
        window_df: pd.DataFrame,
        config: dict,
        mode: str = "train",
        transform: Optional[torch.nn.Module] = None,
    ):
        """Initialize SoundscapeDataset.

        Args:
            window_df: DataFrame from soundscape_windows.csv (filtered by split)
            config: Config dict
            mode: 'train' (labeled only), 'pseudo' (all), 'val' (held-out labeled)
            transform: Optional mel transform
        """
        self.window_df = window_df.reset_index(drop=True)
        self.config = config
        self.mode = mode
        self.transform = transform

        self.sample_rate = config.get("audio", {}).get("sample_rate", 32000)
        self.n_samples = config.get("audio", {}).get("n_samples", 160000)

        self.audio_root = Path(config.get("paths", {}).get("train_soundscapes", "./data/train_soundscapes"))

        # Load taxonomy for species mapping
        taxonomy_path = config.get("paths", {}).get("taxonomy", "./data/taxonomy.csv")
        self.taxonomy_df = pd.read_csv(taxonomy_path)
        self.species_to_idx = {
            str(row["primary_label"]): idx for idx, row in self.taxonomy_df.iterrows()
        }

        print(f"SoundscapeDataset {mode}: {len(self.window_df)} windows")

    def __len__(self) -> int:
        return len(self.window_df)

    def _load_window(self, filename: str, start_sec: float) -> torch.Tensor:
        """Load 5-second window from soundscape file.

        Args:
            filename: Soundscape filename
            start_sec: Start time in seconds

        Returns:
            Waveform [160000]
        """
        if not HAS_LIBROSA:
            raise ImportError("librosa is required for audio loading. Install with: pip install librosa")

        audio_path = self.audio_root / filename

        try:
            # Load with offset (librosa can seek efficiently)
            waveform, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
                offset=start_sec,
                duration=self.n_samples / self.sample_rate,  # 5.0 seconds
            )
        except Exception as e:
            print(f"Error loading {audio_path} at {start_sec}s: {e}")
            return torch.zeros(self.n_samples, dtype=torch.float32)

        # Handle edge case: end of file
        if len(waveform) < self.n_samples:
            # Pad with zeros (acceptable for soundscapes as they naturally end)
            pad_length = self.n_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_length), mode="constant")

        return torch.from_numpy(waveform[:self.n_samples]).float()

    def _parse_labels(self, label_str: str) -> np.ndarray:
        """Parse semicolon-separated species IDs into multi-hot vector.

        Args:
            label_str: "species1;species2;species3" or nan

        Returns:
            Multi-hot label [234]
        """
        label = np.zeros(234, dtype=np.float32)

        if pd.isna(label_str) or not isinstance(label_str, str) or label_str.strip() == "":
            return label

        # Split by semicolon
        species_list = [s.strip() for s in label_str.split(";") if s.strip()]

        for species_id in species_list:
            if species_id in self.species_to_idx:
                idx = self.species_to_idx[species_id]
                label[idx] = 1.0

        return label

    def __getitem__(self, idx: int) -> dict:
        """Get one soundscape window.

        Returns dict with waveform, label, temporal features, metadata.
        """
        row = self.window_df.iloc[idx]

        filename = row["filename"]
        start_sec = float(row["start_sec"])

        # Load audio window
        waveform = self._load_window(filename, start_sec)

        # Parse labels (if available)
        label_str = row.get("primary_label", "")
        label = self._parse_labels(label_str)
        has_label = label.sum() > 0

        # Get temporal features (precomputed in window index)
        hour = float(row.get("hour", 0)) + float(row.get("minute", 0)) / 60.0
        time_sin = float(row.get("hour_sin", np.sin(2 * np.pi * hour / 24)))
        time_cos = float(row.get("hour_cos", np.cos(2 * np.pi * hour / 24)))

        # Build window ID for submission
        window_id = f"{filename}_{int(start_sec)}"

        return {
            "waveform": waveform,
            "label": torch.from_numpy(label).float(),
            "has_label": has_label,
            "hour": hour,
            "time_sin": time_sin,
            "time_cos": time_cos,
            "filename": filename,
            "window_id": window_id,
        }


# Smoke test
if __name__ == "__main__":
    print("Testing datasets...")

    # This is a minimal smoke test - full test requires actual data files
    # Just verify imports and class initialization

    print("✓ Imports successful")
    print("✓ FocalDataset class defined")
    print("✓ SoundscapeDataset class defined")
    print("✓ CORRUPTED_FILES constant defined")
    print("OK")
