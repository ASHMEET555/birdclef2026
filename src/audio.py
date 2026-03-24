"""Audio utilities for BirdCLEF 2026 precompute pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str, sr: int = 32000, mono: bool = True) -> np.ndarray:
    """Load audio file and return float32 waveform at target sample rate."""
    wav, _ = librosa.load(path, sr=sr, mono=mono)
    return wav.astype(np.float32, copy=False)


def tile_pad(wav: np.ndarray, target: int = 160000) -> np.ndarray:
    """Tile-pad waveform to exactly target length without zero-padding."""
    if wav.size == 0:
        return np.zeros(target, dtype=np.float32)
    if wav.shape[0] >= target:
        return wav[:target].astype(np.float32, copy=False)
    n_repeat = int(math.ceil(target / float(wav.shape[0])))
    tiled = np.tile(wav, n_repeat)
    return tiled[:target].astype(np.float32, copy=False)


def load_and_chunk(path: str, sr: int = 32000, chunk_samp: int = 160000) -> list[np.ndarray]:
    """Load one file and split into fixed-size chunks using tile padding."""
    try:
        wav = load_audio(path=path, sr=sr, mono=True)
    except Exception:
        return []

    if wav.size < int(0.5 * sr):
        return []

    if wav.shape[0] <= chunk_samp:
        return [tile_pad(wav, target=chunk_samp)]

    chunks: list[np.ndarray] = []
    n_full = wav.shape[0] // chunk_samp
    for i in range(n_full):
        start = i * chunk_samp
        chunks.append(wav[start : start + chunk_samp].astype(np.float32, copy=False))

    rem = wav.shape[0] % chunk_samp
    if rem > 0:
        chunks.append(tile_pad(wav[-rem:], target=chunk_samp))

    return chunks


def get_audio_duration(path: str) -> float:
    """Fast duration query with soundfile.info (header only)."""
    info = sf.info(path)
    return float(info.duration)


def is_inat_file(filename: str) -> bool:
    """Return True when filename represents iNaturalist recording."""
    return "inat" in Path(filename).name.lower()
