"""BirdCLEF 2026 source package.

This package contains all training, inference, and evaluation code.
Modules are designed to be imported independently to avoid dependency issues.

Usage:
    from src.loss import get_loss
    from src.model import build_model
    from src.train import train
"""

# Don't import anything by default to avoid forcing all dependencies
# Users should import specific modules as needed

__version__ = "0.1.0"

__all__ = []
