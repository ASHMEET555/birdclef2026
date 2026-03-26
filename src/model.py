"""Model definitions for BirdCLEF 2026.

EfficientNet-B0 backbone + GEM pooling + 2-layer classification head.
Confirmed architecture from 2025 2nd place solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized Mean Pooling for 2D feature maps.

    WHY: Standard average pooling treats all frames equally. GEM up-weights
    frames with strong activations (e.g., bird calls) and down-weights
    silence/noise frames. More robust than max pooling which can be
    dominated by outliers.

    Formula: GeM(x) = (mean(x^p))^(1/p)
    - p=1: arithmetic mean (standard avg pooling)
    - p=∞: max pooling
    - p=3: confirmed optimal from 2025 2nd place solution

    p is learnable and initialized to 3.0.

    CONFIRMED: GEM > avg pooling in 2025 2nd place ablation.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        """Initialize GEM pooling.

        Args:
            p: Power parameter (default 3.0, learned during training)
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GEM pooling.

        Args:
            x: Feature map [B, C, H, W]

        Returns:
            Pooled features [B, C]
        """
        # Clamp to avoid numerical issues with small values
        x = torch.clamp(x, min=self.eps)

        # x^p
        x = x.pow(self.p)

        # Mean over spatial dimensions
        x = x.mean(dim=(-2, -1))

        # (mean)^(1/p)
        x = x.pow(1.0 / self.p)

        return x


@dataclass
class ModelConfig:
    """Configuration for BirdCLEF model."""

    num_classes: int = 234
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    gem_p: float = 3.0
    head_dropout: float = 0.25
    use_time_conditioning: bool = False
    in_channels: int = 1


class BirdCLEFModel(nn.Module):
    """EfficientNet-B0 + GEM + 2-layer head for BirdCLEF 2026.

    Architecture:
        Input: [B, 1, 128, 313] mel spectrogram
        ↓
        EfficientNet-B0 backbone (modified first conv for 1-channel input)
        ↓
        GEM pooling → [B, 1280]
        ↓
        Optional: concatenate time encoding [B, 2] → [B, 1282]
        ↓
        Linear(1280 or 1282 → 512) → ReLU → Dropout(0.25)
        ↓
        Linear(512 → 234)
        ↓
        Output: logits [B, 234] and probs [B, 234]

    WHY 2-layer head: Single linear layer underfits. 2-layer head with ReLU
    allows non-linear decision boundaries in embedding space.
    Confirmed from 2025 2nd place solution.
    """

    def __init__(self, config: ModelConfig):
        """Initialize BirdCLEF model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Load backbone from timm
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,  # Remove classifier
            global_pool="",  # Remove built-in pooling
        )

        # Modify first conv for 1-channel input (if needed)
        if config.in_channels != 3:
            self._modify_first_conv(config.in_channels)

        # GEM pooling
        self.gem = GeM(p=config.gem_p)

        # Get backbone output dimension
        # For EfficientNet-B0, this is 1280
        with torch.no_grad():
            dummy_input = torch.randn(1, config.in_channels, 128, 313)
            backbone_out = self.backbone(dummy_input)
            pooled = self.gem(backbone_out)
            self.backbone_dim = pooled.shape[1]

        # Time conditioning adds 2 features (sin, cos)
        head_input_dim = self.backbone_dim
        if config.use_time_conditioning:
            head_input_dim += 2

        # 2-layer classification head
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.head_dropout),
            nn.Linear(512, config.num_classes),
        )

    def _modify_first_conv(self, in_channels: int):
        """Modify first conv layer to accept different number of input channels.

        If pretrained, averages RGB weights to single channel.
        """
        if hasattr(self.backbone, "conv_stem"):
            # EfficientNet naming
            old_conv = self.backbone.conv_stem
        elif hasattr(self.backbone, "features") and hasattr(self.backbone.features[0], "0"):
            # Alternative naming
            old_conv = self.backbone.features[0][0]
        else:
            print("Warning: Could not find first conv layer to modify")
            return

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        # Copy pretrained weights if available
        if self.config.pretrained and in_channels == 1:
            with torch.no_grad():
                # Average RGB weights to create single-channel weights
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

        # Replace first conv
        if hasattr(self.backbone, "conv_stem"):
            self.backbone.conv_stem = new_conv
        elif hasattr(self.backbone, "features"):
            self.backbone.features[0][0] = new_conv

    def forward(
        self,
        mel: torch.Tensor,
        time_encoding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            mel: Mel spectrogram [B, 1, 128, 313]
            time_encoding: Optional time encoding [B, 2] with (sin, cos)

        Returns:
            Tuple of (logits, probs):
                logits: [B, num_classes] raw outputs for loss computation
                probs: [B, num_classes] sigmoid probabilities for inference
        """
        # Backbone features
        features = self.backbone(mel)  # [B, C, H, W]

        # GEM pooling
        pooled = self.gem(features)  # [B, backbone_dim]

        # Optionally concatenate time encoding
        if time_encoding is not None and self.config.use_time_conditioning:
            pooled = torch.cat([pooled, time_encoding], dim=1)  # [B, backbone_dim + 2]

        # Classification head
        logits = self.head(pooled)  # [B, num_classes]

        # Sigmoid for probabilities
        probs = torch.sigmoid(logits)

        return logits, probs

    def get_embedding(
        self,
        mel: torch.Tensor,
        time_encoding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract embeddings (before classification head).

        Useful for visualization, retrieval, and pseudo-labeling.

        Args:
            mel: Mel spectrogram [B, 1, 128, 313]
            time_encoding: Optional time encoding [B, 2]

        Returns:
            Embeddings [B, backbone_dim] or [B, backbone_dim + 2]
        """
        features = self.backbone(mel)
        pooled = self.gem(features)

        if time_encoding is not None and self.config.use_time_conditioning:
            pooled = torch.cat([pooled, time_encoding], dim=1)

        return pooled


def build_model(
    num_classes: int = 234,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    gem_p: float = 3.0,
    head_dropout: float = 0.25,
    use_time_conditioning: bool = False,
    in_channels: int = 1,
) -> BirdCLEFModel:
    """Factory function to build model from config.

    Args:
        num_classes: Number of species (default 234)
        backbone: Backbone architecture (default "efficientnet_b0")
        pretrained: Use ImageNet pretrained weights (default True)
        gem_p: GEM pooling power parameter (default 3.0)
        head_dropout: Dropout rate in head (default 0.25)
        use_time_conditioning: Whether to use time-of-day features (default False)
        in_channels: Number of input channels (default 1)

    Returns:
        BirdCLEFModel instance
    """
    config = ModelConfig(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        gem_p=gem_p,
        head_dropout=head_dropout,
        use_time_conditioning=use_time_conditioning,
        in_channels=in_channels,
    )

    return BirdCLEFModel(config)


# Smoke test
if __name__ == "__main__":
    print("Testing model...")

    # Create model
    model = build_model(
        num_classes=234,
        pretrained=False,  # Skip pretrained for faster test
        use_time_conditioning=False,
    )

    print(f"Model created: {model.__class__.__name__}")
    print(f"Backbone dim: {model.backbone_dim}")

    # Test forward pass
    batch_size = 2
    mel = torch.randn(batch_size, 1, 128, 313)

    logits, probs = model(mel)

    print(f"Input shape: {mel.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Probs range: [{probs.min():.3f}, {probs.max():.3f}]")

    assert logits.shape == (batch_size, 234), f"Unexpected logits shape: {logits.shape}"
    assert probs.shape == (batch_size, 234), f"Unexpected probs shape: {probs.shape}"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probs not in [0, 1]"

    # Test with time conditioning
    model_with_time = build_model(
        num_classes=234,
        pretrained=False,
        use_time_conditioning=True,
    )

    time_encoding = torch.randn(batch_size, 2)  # (sin, cos)
    logits, probs = model_with_time(mel, time_encoding)

    print(f"With time conditioning:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probs shape: {probs.shape}")

    # Test embedding extraction
    embeddings = model.get_embedding(mel)
    print(f"Embeddings shape: {embeddings.shape}")

    print("✓ All model tests passed")
    print("OK")
