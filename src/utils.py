"""Utility functions for BirdCLEF 2026 precompute pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def flush_perch_batch(perch_model, chunk_buffer: list[np.ndarray], perch_batch_size: int = 64) -> np.ndarray:
	"""Run fixed-size Perch inference and return embeddings for real chunks only."""
	n_real = len(chunk_buffer)
	if n_real == 0:
		return np.empty((0, 1280), dtype=np.float32)
	if n_real > perch_batch_size:
		raise ValueError(f"chunk_buffer too large ({n_real}) > perch_batch_size ({perch_batch_size})")

	batch = np.zeros((perch_batch_size, chunk_buffer[0].shape[0]), dtype=np.float32)
	for i, chunk in enumerate(chunk_buffer):
		batch[i] = chunk

	result = perch_model.infer_tf(tf.convert_to_tensor(batch, dtype=tf.float32))
	emb = result["embedding"].numpy().astype(np.float32, copy=False)
	return emb[:n_real]


def warmup_perch(
	perch_model,
	perch_batch_size: int = 64,
	chunk_samp: int = 160000,
	n_passes: int = 5,
) -> None:
	"""Warm up Perch/XLA kernels using fixed-size zero batches."""
	warm = np.zeros((perch_batch_size, chunk_samp), dtype=np.float32)
	t = tf.convert_to_tensor(warm, dtype=tf.float32)
	for _ in range(n_passes):
		_ = perch_model.infer_tf(t)


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
