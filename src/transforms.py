"""Audio feature transforms and augmentations for BirdCLEF 2026.

All augmentations are audio-domain (applied to waveform BEFORE mel conversion)
except SpecAugment which is applied to mel spectrogram.

CONFIRMED CONFIG: All parameters from Phase 4 EDA + 2025 2nd place ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional dependencies - lazy import to avoid hard requirement
try:
    import torchaudio
    import torchaudio.functional as AF
    HAS_TORCHAUDIO = True
    # Check if PCEN is available (newer torchaudio versions)
    try:
        HAS_PCEN = hasattr(torchaudio.transforms, 'PCEN')
    except:
        HAS_PCEN = False
except (ImportError, OSError, Exception) as e:
    # Handle all import errors including CUDA compatibility issues
    HAS_TORCHAUDIO = False
    HAS_PCEN = False

class TorchPCEN(nn.Module):
    """Torch-based PCEN fallback to keep computation on GPU.

    Implements the standard PCEN formula with an IIR smoother over time.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        s: float = 0.025,
        alpha: float = 0.98,
        delta: float = 2.0,
        r: float = 0.5,
    ):
        super().__init__()
        self.eps = eps
        self.s = s
        self.alpha = alpha
        self.delta = delta
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PCEN to a mel spectrogram.

        Args:
            x: [B, n_mels, time]

        Returns:
            PCEN-normalized tensor [B, n_mels, time]
        """
        # IIR smoothing along time dimension
        b, m, t = x.shape
        m_t = torch.zeros((b, m), device=x.device, dtype=x.dtype)
        out = []
        for i in range(t):
            m_t = (1.0 - self.s) * m_t + self.s * x[:, :, i]
            smooth = m_t
            pcen = (x[:, :, i] / (self.eps + smooth).pow(self.alpha) + self.delta).pow(self.r) - self.delta**self.r
            out.append(pcen)

        return torch.stack(out, dim=2)


@dataclass
class MelSpecConfig:
    """Configuration for mel spectrogram extraction.

    NEVER change these without explicit instruction - confirmed from EDA.
    """

    sample_rate: int = 32000
    chunk_duration: float = 5.0  # seconds
    n_samples: int = 160000  # 5 * 32000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 20.0  # NOT 50 - captures frog fundamentals at 200-500Hz
    f_max: float = 16000.0  # Nyquist for 32kHz
    top_db: float = 80.0
    power: float = 2.0
    use_pcen: bool = True  # PCEN normalization - DO NOT REMOVE (-0.049 val AUC cost)


class DummyMelTransform(nn.Module):
    """Fallback transform when torchaudio is not available.

    Returns random mel spectrograms for testing purposes only.
    DO NOT use for actual training - torchaudio required for real work.
    """

    def __init__(self, config: MelSpecConfig):
        super().__init__()
        self.config = config
        self.n_mels = config.n_mels
        # Calculate output time dimension: (n_samples / hop_length) + 1
        self.time_dim = (config.n_samples // config.hop_length) + 1

    def forward(self, x):
        # Return dummy mel spectrogram with correct shape and dtype for testing
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        device = x.device if hasattr(x, 'device') else 'cpu'
        dtype = x.dtype if hasattr(x, 'dtype') else torch.float32
        return torch.randn(batch_size, 1, self.n_mels, self.time_dim,
                          device=device, dtype=dtype)

    def to(self, device):
        return self


class MelPCENTransform(nn.Module):
    """Convert raw waveform to mel spectrogram with PCEN normalization.

    Output shape: [1, 128, 313] for a 5-second chunk at 32kHz

    WHY PCEN: Per-Channel Energy Normalization adaptively compresses dynamic range
    and suppresses stationary background noise. Essential for handling variable
    recording quality across XC (SNR 10.3dB) and iNat (SNR 7.1dB) sources.

    Removing PCEN costs -0.049 val AUC (confirmed ablation).
    """

    def __init__(self, config: MelSpecConfig):
        super().__init__()
        self.config = config

        if not HAS_TORCHAUDIO:
            raise ImportError("torchaudio is required for mel transforms. Install with: pip install torchaudio")

        # Mel spectrogram transform
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=config.power,
        )

        # PCEN normalization - KEEP IN FP32 ALWAYS (never quantize this layer)
        if config.use_pcen and HAS_PCEN:
            self.pcen = torchaudio.transforms.PCEN(
                eps=1e-6,
                s=0.025,
                alpha=0.98,
                delta=2.0,
                r=0.5,
            )
            self.use_torch_pcen = False
        else:
            if config.use_pcen and not HAS_PCEN:
                print("Warning: torchaudio PCEN not available, falling back to torch PCEN")
            self.pcen = None
            self.use_torch_pcen = config.use_pcen
            if self.use_torch_pcen:
                self.torch_pcen = TorchPCEN(
                    eps=1e-6,
                    s=0.025,
                    alpha=0.98,
                    delta=2.0,
                    r=0.5,
                )
            # Fallback to log-mel with amplitude to dB
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
                stype="power", top_db=config.top_db
            )
        self._warned_torch_pcen = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel features.

        Args:
            waveform: [B, n_samples] or [n_samples] mono audio at 32kHz

        Returns:
            mel: [B, 1, n_mels, time_frames] or [1, n_mels, time_frames]
                 Shape: [*, 1, 128, 313] for 5-second chunks
        """
        # Ensure waveform has batch dimension
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [1, n_samples]

        # Compute mel spectrogram: [B, n_mels, time_frames]
        mel = self.melspec(waveform)

        # Apply normalization
        if self.pcen is not None:
            # PCEN normalization
            mel = self.pcen(mel)
        elif self.use_torch_pcen:
            if not self._warned_torch_pcen:
                print("Using torch PCEN fallback (torchaudio PCEN unavailable)")
                self._warned_torch_pcen = True
            mel = self.torch_pcen(mel)
        else:
            # Log-mel fallback
            mel = self.amplitude_to_db(mel)

        # Add channel dimension: [B, 1, n_mels, time_frames]
        mel = mel.unsqueeze(1)

        return mel


class AudioAugmentation:
    """Audio-domain augmentations applied to raw waveform BEFORE mel conversion.

    All methods are static and stateless. Apply selectively during training only.

    CONFIRMED: Audio-domain MixUp = +0.036 private LB (2025 2nd place)
    NEVER implement spectrogram-level MixUp - it is inferior.
    """

    @staticmethod
    def mixup(
        waveform_a: torch.Tensor,
        label_a: torch.Tensor,
        waveform_b: torch.Tensor,
        label_b: torch.Tensor,
        alpha: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Audio-domain MixUp augmentation.

        WHY: Bridges domain gap between clean XC clips and noisy soundscapes.
        Mixing at waveform level preserves phase relationships and creates
        realistic multi-species scenarios.

        CONFIRMED: +0.036 private LB improvement (2025 2nd place)
        DO NOT use spectrogram-level MixUp - confirmed inferior in ablations.

        Args:
            waveform_a: [n_samples] first audio
            waveform_b: [n_samples] second audio
            label_a: [num_classes] first label vector
            label_b: [num_classes] second label vector
            alpha: Beta distribution parameter (default 0.5)

        Returns:
            Tuple of (mixed_waveform, mixed_label)
                mixed_waveform: [n_samples] weighted audio mix
                mixed_label: [num_classes] element-wise max (NOT weighted sum)

        Notes:
            - Lambda sampled from Beta(alpha, alpha) distribution
            - Label mixing uses element-wise MAX, not weighted sum
              (confirmed from 2025 2nd place - handles multi-label better)
        """
        # Sample mixing weight from Beta distribution
        if alpha > 0:
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
        else:
            lam = 1.0

        # Mix waveforms
        mixed_waveform = lam * waveform_a + (1 - lam) * waveform_b

        # Mix labels: element-wise max (2025 2nd place approach)
        # WHY: If either mixture component has a species, the mix has that species
        # More robust for multi-label than weighted sum which can dilute weak signals
        mixed_label = torch.maximum(label_a, label_b)

        return mixed_waveform, mixed_label

    @staticmethod
    def random_gain(
        waveform: torch.Tensor,
        gain_db: float = 6.0,
        prob: float = 1.0,
    ) -> torch.Tensor:
        """Apply random gain between -gain_db and +gain_db.

        WHY: Bridges SNR gap between XC (10.3dB SNR) and iNat (7.1dB SNR).
        Model must be robust to varying recording levels.

        Args:
            waveform: [n_samples] audio
            gain_db: Maximum gain in dB (default ±6dB)
            prob: Probability of applying (default 1.0)

        Returns:
            Augmented waveform
        """
        if torch.rand(1).item() > prob:
            return waveform

        # Sample random gain in dB
        gain_db_val = torch.empty(1).uniform_(-gain_db, gain_db).item()

        # Convert dB to amplitude ratio: ratio = 10^(dB/20)
        gain_ratio = 10.0 ** (gain_db_val / 20.0)

        # Apply gain and clip to [-1, 1]
        augmented = waveform * gain_ratio
        augmented = torch.clamp(augmented, -1.0, 1.0)

        return augmented

    @staticmethod
    def random_filtering(
        waveform: torch.Tensor,
        sr: int = 32000,
        prob: float = 0.5,
    ) -> torch.Tensor:
        """Apply random biquad EQ filter to simulate different microphones.

        WHY: Different recording devices have different frequency responses.
        RandomFiltering teaches model to be invariant to microphone characteristics.

        CONFIRMED: +~1% private LB improvement (2025 2nd place)

        Args:
            waveform: [n_samples] audio
            sr: Sample rate (default 32000)
            prob: Probability of applying (default 0.5)

        Returns:
            Filtered waveform

        Notes:
            - Randomly chooses between highpass, lowpass, bandpass filters
            - Random center frequency and Q factor
            - Simulates realistic microphone frequency responses
        """
        if not HAS_TORCHAUDIO:
            # Silently skip if torchaudio not available
            return waveform

        if torch.rand(1).item() > prob:
            return waveform

        # Randomly choose filter type
        filter_type = torch.randint(0, 3, (1,)).item()

        try:
            if filter_type == 0:
                # Highpass filter: 100-1000 Hz cutoff
                cutoff = torch.empty(1).uniform_(100, 1000).item()
                filtered = AF.highpass_biquad(waveform, sr, cutoff)
            elif filter_type == 1:
                # Lowpass filter: 8000-14000 Hz cutoff
                cutoff = torch.empty(1).uniform_(8000, 14000).item()
                filtered = AF.lowpass_biquad(waveform, sr, cutoff)
            else:
                # Bandpass filter: random center freq 500-4000 Hz, Q=0.5-2.0
                center_freq = torch.empty(1).uniform_(500, 4000).item()
                q_factor = torch.empty(1).uniform_(0.5, 2.0).item()
                filtered = AF.bandpass_biquad(waveform, sr, center_freq, q_factor)
        except Exception:
            # Fallback if biquad functions not available
            filtered = waveform

        return filtered

    @staticmethod
    def add_background_noise(
        waveform: torch.Tensor,
        noise_waveform: torch.Tensor,
        snr_db_min: float = 5.0,
        snr_db_max: float = 15.0,
        prob: float = 0.5,
    ) -> torch.Tensor:
        """Add background noise from soundscape at random SNR.

        WHY: Further bridges domain gap between clean clips and noisy soundscapes.
        Used in Stage 3+ training after pseudo-labeling unlabeled soundscapes.

        Args:
            waveform: [n_samples] clean audio signal
            noise_waveform: [n_samples] background noise
            snr_db_min: Minimum SNR in dB (default 5)
            snr_db_max: Maximum SNR in dB (default 15)
            prob: Probability of applying (default 0.5)

        Returns:
            Signal + noise at random SNR

        Notes:
            - SNR = 10 log10(P_signal / P_noise)
            - Higher SNR = cleaner signal, lower SNR = noisier
        """
        if torch.rand(1).item() > prob:
            return waveform

        # Ensure same length
        if len(noise_waveform) < len(waveform):
            # Tile noise to match signal length
            n_repeats = (len(waveform) // len(noise_waveform)) + 1
            noise_waveform = noise_waveform.repeat(n_repeats)[: len(waveform)]
        elif len(noise_waveform) > len(waveform):
            # Random crop noise
            start_idx = torch.randint(0, len(noise_waveform) - len(waveform) + 1, (1,)).item()
            noise_waveform = noise_waveform[start_idx : start_idx + len(waveform)]

        # Sample random SNR
        snr_db = torch.empty(1).uniform_(snr_db_min, snr_db_max).item()

        # Compute signal and noise power
        signal_power = waveform.pow(2).mean()
        noise_power = noise_waveform.pow(2).mean()

        # Compute noise scaling factor to achieve target SNR
        # SNR_dB = 10 * log10(P_signal / P_noise)
        # P_noise_target = P_signal / (10^(SNR_dB / 10))
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-8))

        # Add scaled noise
        noisy_waveform = waveform + noise_scale * noise_waveform

        # Clip to valid range
        noisy_waveform = torch.clamp(noisy_waveform, -1.0, 1.0)

        return noisy_waveform


class SpecAugment(nn.Module):
    """Time and frequency masking on mel spectrogram.

    Apply AFTER mel conversion, BEFORE PCEN normalization for best results.

    WHY: Forces model to not rely on specific time frames or frequency bands.
    Improves robustness to partial occlusions and missing data.

    CONFIRMED: Positive contribution in 2025 2nd place ablation.
    """

    def __init__(
        self,
        time_mask_max: int = 40,
        freq_mask_max: int = 20,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ):
        """Initialize SpecAugment.

        Args:
            time_mask_max: Maximum width of time masks (in frames)
            freq_mask_max: Maximum width of frequency masks (in mel bins)
            num_time_masks: Number of time masks to apply
            num_freq_masks: Number of frequency masks to apply
        """
        super().__init__()
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to mel spectrogram.

        Args:
            mel: [B, 1, n_mels, time_frames] mel spectrogram

        Returns:
            Augmented mel spectrogram with same shape
        """
        # Only apply during training
        if not self.training:
            return mel

        batch_size, channels, n_mels, time_frames = mel.shape

        # Clone to avoid in-place modification
        mel_aug = mel.clone()

        for b in range(batch_size):
            # Apply frequency masks
            for _ in range(self.num_freq_masks):
                freq_width = torch.randint(0, self.freq_mask_max, (1,)).item()
                if freq_width > 0 and n_mels > freq_width:
                    freq_start = torch.randint(0, n_mels - freq_width, (1,)).item()
                    mel_aug[b, :, freq_start : freq_start + freq_width, :] = 0.0

            # Apply time masks
            for _ in range(self.num_time_masks):
                time_width = torch.randint(0, self.time_mask_max, (1,)).item()
                if time_width > 0 and time_frames > time_width:
                    time_start = torch.randint(0, time_frames - time_width, (1,)).item()
                    mel_aug[b, :, :, time_start : time_start + time_width] = 0.0

        return mel_aug


def build_mel_transform(config: dict) -> nn.Module:
    """Factory function to build mel spectrogram transform from config.

    Args:
        config: Config dict with 'audio' section

    Returns:
        MelPCENTransform if torchaudio available, DummyMelTransform otherwise
    """
    audio_config = config.get("audio", {})

    mel_config = MelSpecConfig(
        sample_rate=audio_config.get("sample_rate", 32000),
        chunk_duration=audio_config.get("chunk_duration", 5.0),
        n_samples=audio_config.get("n_samples", 160000),
        n_fft=audio_config.get("n_fft", 2048),
        hop_length=audio_config.get("hop_length", 512),
        n_mels=audio_config.get("n_mels", 128),
        f_min=audio_config.get("f_min", 20.0),
        f_max=audio_config.get("f_max", 16000.0),
        top_db=audio_config.get("top_db", 80.0),
        use_pcen=True,  # Always use PCEN unless explicitly disabled
    )

    if HAS_TORCHAUDIO:
        return MelPCENTransform(mel_config)
    else:
        print("⚠ Warning: torchaudio not available, using dummy transform for testing only")
        return DummyMelTransform(mel_config)


# Smoke test
if __name__ == "__main__":
    print("Testing transforms...")

    # Create dummy waveform
    sr = 32000
    duration = 5.0
    n_samples = int(sr * duration)
    waveform = torch.randn(n_samples)

    # Test Mel+PCEN transform
    config = {
        "audio": {
            "sample_rate": sr,
            "chunk_duration": duration,
            "n_samples": n_samples,
            "n_mels": 128,
            "f_min": 20.0,
            "f_max": 16000.0,
        }
    }

    mel_transform = build_mel_transform(config)
    mel = mel_transform(waveform)
    print(f"Mel shape: {mel.shape} (expected [1, 1, 128, 313])")
    assert mel.shape == (1, 1, 128, 313), f"Unexpected mel shape: {mel.shape}"

    # Test audio augmentations
    waveform_a = torch.randn(n_samples)
    waveform_b = torch.randn(n_samples)
    label_a = torch.zeros(234)
    label_b = torch.zeros(234)
    label_a[10] = 1.0
    label_b[20] = 1.0

    # Test MixUp
    mixed_wave, mixed_label = AudioAugmentation.mixup(
        waveform_a, label_a, waveform_b, label_b, alpha=0.5
    )
    print(f"MixUp: waveform shape {mixed_wave.shape}, label sum {mixed_label.sum().item()}")
    assert mixed_wave.shape == waveform_a.shape
    assert mixed_label.sum().item() == 2.0  # Element-wise max of two one-hot

    # Test random gain
    gained = AudioAugmentation.random_gain(waveform, gain_db=6.0)
    print(f"Random gain: original range [{waveform.min():.3f}, {waveform.max():.3f}], "
          f"gained range [{gained.min():.3f}, {gained.max():.3f}]")
    assert gained.shape == waveform.shape

    # Test random filtering
    filtered = AudioAugmentation.random_filtering(waveform, sr=sr, prob=1.0)
    print(f"Random filtering: shape {filtered.shape}")
    assert filtered.shape == waveform.shape

    # Test background noise
    noise = torch.randn(n_samples) * 0.1
    noisy = AudioAugmentation.add_background_noise(waveform, noise, prob=1.0)
    print(f"Background noise: shape {noisy.shape}")
    assert noisy.shape == waveform.shape

    # Test SpecAugment
    spec_aug = SpecAugment(time_mask_max=40, freq_mask_max=20, num_time_masks=2, num_freq_masks=2)
    spec_aug.train()  # Set to training mode
    mel_aug = spec_aug(mel)
    print(f"SpecAugment: shape {mel_aug.shape}")
    assert mel_aug.shape == mel.shape

    print("✓ All transforms work correctly")
    print("OK")
