"""Utility functions for BirdCLEF 2026 precompute pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_output_dirs(base_out: Path) -> None:
	"""Create full output directory tree for precompute artifacts."""
	dirs = [
		base_out,
		base_out / "metadata",
		base_out / "label_vectors",
		base_out / "perch_embeddings",
		base_out / "perch_embeddings" / "train_audio",
		base_out / "perch_embeddings" / "soundscapes",
	]
	for d in dirs:
		d.mkdir(parents=True, exist_ok=True)


def get_embedding_path(out_dir: Path, species_id: str, audio_filename: str) -> Path:
	"""Build train-audio embedding path for one clip."""
	stem = Path(audio_filename).stem
	return out_dir / "perch_embeddings" / "train_audio" / species_id / f"{stem}.npy"


def load_embedding(emb_path: str) -> dict:
	"""Load saved embedding dictionary from .npy file."""
	return np.load(emb_path, allow_pickle=True).item()


def resume_filter(df: pd.DataFrame, out_dir: Path) -> list[int]:
	"""Return row indices whose expected train embedding file does not exist."""
	missing: list[int] = []
	for idx, row in df.iterrows():
		rel = Path(str(row["filename"]))
		species_id = rel.parts[0]
		emb_path = get_embedding_path(out_dir=out_dir, species_id=species_id, audio_filename=rel.name)
		if not emb_path.exists():
			missing.append(int(idx))
	return missing
