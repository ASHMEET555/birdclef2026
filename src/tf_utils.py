"""TensorFlow-specific utilities for Perch precompute."""

from __future__ import annotations

import numpy as np


def flush_perch_batch(perch_model, chunk_buffer: list[np.ndarray], perch_batch_size: int = 64) -> np.ndarray:
    import tensorflow as tf

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
    import tensorflow as tf

    """Warm up Perch/XLA kernels using fixed-size zero batches."""
    warm = np.zeros((perch_batch_size, chunk_samp), dtype=np.float32)
    t = tf.convert_to_tensor(warm, dtype=tf.float32)
    for _ in range(n_passes):
        _ = perch_model.infer_tf(t)
