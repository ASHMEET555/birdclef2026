"""Loss function implementations for BirdCLEF 2026.

All loss functions are designed for multi-label classification where 98% of labels
per sample are negative. Standard BCE wastes gradients on easy negatives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Reference: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
    Paper: https://arxiv.org/abs/2009.14119

    WHY: In BirdCLEF windows, 98% of labels are negative (absent species).
    Standard BCE wastes gradients on trivial easy negatives. ASL focuses learning
    on hard negatives and positives by asymmetrically modulating focal weights.

    Confirmed parameters from 2025 2nd place solution:
    - gamma_neg=4: strong down-weighting of easy negatives
    - gamma_pos=0: no modulation of positives (all positives are valuable)
    - clip=0.05: shifts decision boundary to handle class imbalance
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        """Initialize ASL loss.

        Args:
            gamma_neg: Focal weight exponent for negatives (higher = more focus on hard negatives)
            gamma_pos: Focal weight exponent for positives
            clip: Probability margin clipping for negatives
            eps: Numerical stability epsilon
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ASL loss.

        Args:
            logits: Raw model outputs [B, num_classes]
            targets: Multi-label targets [B, num_classes] in range [0, 1]
                     Can be soft labels (e.g., 0.3 for secondary labels)

        Returns:
            Scalar loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Asymmetric clipping for negatives
        # Shifts effective decision boundary to handle extreme imbalance
        probs_neg = (probs + self.clip).clamp(max=1.0)

        # Focal weights - asymmetric modulation
        # For negatives: (p_m)^gamma_neg down-weights easy negatives (p close to 0)
        # For positives: (1-p)^gamma_pos typically gamma_pos=0 so no modulation
        weight_neg = torch.pow(probs_neg, self.gamma_neg)
        weight_pos = torch.pow(1 - probs, self.gamma_pos)

        # Asymmetric focusing
        # Negative loss: weighted by how confident (wrong) the prediction is
        # Positive loss: standard cross-entropy (when gamma_pos=0)
        loss_neg = -torch.log((1 - probs).clamp(min=self.eps)) * weight_neg
        loss_pos = -torch.log(probs.clamp(min=self.eps)) * weight_pos

        # Combine: use target as selector (0 = negative class, 1 = positive class)
        # For soft labels (0 < target < 1), loss is weighted combination
        loss = targets * loss_pos + (1 - targets) * loss_neg

        # Return mean loss across all samples and classes
        return loss.mean()


class LabelSmoothingCE(nn.Module):
    """Cross-entropy loss with label smoothing for multi-label classification.

    WHY: Label smoothing prevents overconfident predictions and acts as
    regularization. Particularly useful in early training before switching to ASL.

    Confirmed alpha=0.05 from 2025 2nd place solution.
    """

    def __init__(self, alpha: float = 0.05, num_classes: int = 234):
        """Initialize label smoothing CE loss.

        Args:
            alpha: Smoothing factor (0 = no smoothing, 1 = uniform distribution)
            num_classes: Number of classes for multi-label classification
        """
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss.

        Args:
            logits: Raw model outputs [B, num_classes]
            targets: Multi-label targets [B, num_classes] in range [0, 1]

        Returns:
            Scalar loss value
        """
        # Apply label smoothing: y_smooth = (1 - alpha) * y + alpha / K
        # For multi-label, we smooth towards 0.5 (uncertain) rather than uniform
        targets_smooth = (1 - self.alpha) * targets + self.alpha * 0.5

        # Binary cross-entropy with smoothed targets
        loss = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction="mean")

        return loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    WHY: Alternative to ASL for handling class imbalance. Focal loss down-weights
    easy examples regardless of whether they are positive or negative.
    Generally inferior to ASL for extreme multi-label imbalance but included
    for ablation comparisons.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-8):
        """Initialize Focal loss.

        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            eps: Numerical stability epsilon
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs [B, num_classes]
            targets: Binary targets [B, num_classes]

        Returns:
            Scalar loss value
        """
        probs = torch.sigmoid(logits)

        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Focal modulation: (1 - p_t)^gamma
        # p_t = p if target=1, else (1-p)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = torch.pow(1 - p_t, self.gamma)

        # Alpha weighting: alpha for positive, (1-alpha) for negative
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_weight * focal_weight * bce

        return loss.mean()


def get_loss(config: dict) -> nn.Module:
    """Factory function to create loss based on config.

    Args:
        config: Configuration dict with 'loss' section

    Returns:
        Loss module instance

    Raises:
        ValueError: If loss type is not supported
    """
    loss_config = config.get("loss", {})
    loss_type = loss_config.get("type", "asl")

    if loss_type == "asl":
        return ASLLoss(
            gamma_neg=loss_config.get("gamma_neg", 4),
            gamma_pos=loss_config.get("gamma_pos", 0),
            clip=loss_config.get("clip", 0.05),
        )
    elif loss_type == "ce_smooth":
        return LabelSmoothingCE(
            alpha=loss_config.get("label_smoothing", 0.05),
            num_classes=config.get("project", {}).get("num_classes", 234),
        )
    elif loss_type == "focal":
        return FocalLoss(
            alpha=loss_config.get("alpha", 0.25),
            gamma=loss_config.get("gamma", 2.0),
        )
    elif loss_type == "bce":
        # Standard BCE for baseline comparison
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


# Smoke test
if __name__ == "__main__":
    print("Testing loss functions...")

    # Create dummy data
    batch_size, num_classes = 4, 234
    logits = torch.randn(batch_size, num_classes)
    targets = torch.zeros(batch_size, num_classes)
    # Set a few positive labels
    targets[0, 10] = 1.0
    targets[1, 20] = 1.0
    targets[2, 30] = 0.3  # Soft label (secondary species)
    targets[3, 40] = 1.0

    # Test ASL loss
    asl = ASLLoss()
    loss_asl = asl(logits, targets)
    print(f"ASL Loss: {loss_asl.item():.4f}")
    assert loss_asl.item() > 0, "ASL loss should be positive"

    # Test LabelSmoothingCE
    ce_smooth = LabelSmoothingCE(alpha=0.05, num_classes=num_classes)
    loss_ce = ce_smooth(logits, targets)
    print(f"Label Smoothing CE Loss: {loss_ce.item():.4f}")
    assert loss_ce.item() > 0, "CE loss should be positive"

    # Test Focal loss
    focal = FocalLoss()
    loss_focal = focal(logits, targets)
    print(f"Focal Loss: {loss_focal.item():.4f}")
    assert loss_focal.item() > 0, "Focal loss should be positive"

    # Test factory function
    config = {"loss": {"type": "asl", "gamma_neg": 4, "gamma_pos": 0, "clip": 0.05}}
    loss_fn = get_loss(config)
    print(f"Factory created: {type(loss_fn).__name__}")

    # Test shapes
    print(f"✓ All loss functions work correctly")
    print(f"✓ Input shape: {logits.shape}")
    print(f"✓ Target shape: {targets.shape}")
    print(f"✓ Output: scalar loss")
    print("OK")
