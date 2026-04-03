"""Training and validation loops for BirdCLEF 2026.

Main entry point: python src/train.py --config configs/stage2_base.yaml --fold 0

Handles complete training pipeline:
- Mixed precision (AMP) for faster training
- Gradient clipping for stability
- MixUp augmentation at batch level
- Per-class AUC validation metrics
- Checkpoint management
- Experiment logging
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Optional dependencies
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Other imports that might not be available
try:
    import torch.amp as amp
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from tqdm import tqdm
    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False

try:
    # Try relative imports first (when imported as module)
    from .dataset import FocalDataset, SoundscapeDataset
    from .loss import get_loss
    from .metrics import compute_macro_auc, compute_per_class_auc
    from .model import build_model
    from .transforms import AudioAugmentation, build_mel_transform
except ImportError:
    # Fall back to direct imports (when running as script)
    from dataset import FocalDataset, SoundscapeDataset
    from loss import get_loss
    from metrics import compute_macro_auc, compute_per_class_auc
    from model import build_model
    from transforms import AudioAugmentation, build_mel_transform


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML config file with optional base config inheritance.

    Args:
        config_path: Path to YAML config file

    Returns:
        Merged config dictionary
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required for config loading. Install with: pip install pyyaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle base config inheritance (_base key) - recursive
    if "_base" in config:
        base_path = Path(config_path).parent / config["_base"]
        # Recursively load base config (handles multi-level inheritance)
        base_config = load_config(str(base_path))

        # Merge: stage config overrides base config
        merged = {**base_config, **config}
        # Deep merge nested dicts
        for key in base_config:
            if isinstance(base_config[key], dict) and key in config:
                if isinstance(config[key], dict):
                    merged[key] = {**base_config[key], **config[key]}

        config = merged
        # Remove _base key from final config
        config.pop("_base", None)

    return config


def get_device_settings(device: torch.device) -> dict:
    """Get device-specific settings for training.

    Args:
        device: torch device (cuda/cpu)

    Returns:
        Dict with device-specific settings
    """
    is_cuda = device.type == 'cuda'
    return {
        'pin_memory': is_cuda,  # Only use pin_memory with CUDA
        'use_amp': is_cuda,     # Only use mixed precision with CUDA
        'device_type': 'cuda' if is_cuda else 'cpu'
    }


def get_dataloaders(
    config: dict,
    fold: int,
    mel_transform: nn.Module,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders.

    Supports multiple modes:
    - Full training: Use all data
    - Debug quick_test: 50 samples, focal validation only
    - Debug full_pipeline: 2000 samples, optional soundscape validation

    Args:
        config: Config dictionary
        fold: Fold number for cross-validation
        mel_transform: Mel spectrogram transform
        device: Torch device

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError("Training dependencies not available. Install with: pip install torch torchaudio tqdm")

    # Get device-specific settings
    device_settings = get_device_settings(device)

    precomputed_dir = config["paths"]["precomputed"]
    train_folds_path = Path(precomputed_dir) / "metadata" / "train_folds.csv"

    # Load training metadata
    train_folds_df = pd.read_csv(train_folds_path)

    # Filter corrupted files
    try:
        from .dataset import CORRUPTED_FILES
    except ImportError:
        from dataset import CORRUPTED_FILES
    train_folds_df = train_folds_df[~train_folds_df["filename"].isin(CORRUPTED_FILES)]

    # Handle debug modes
    debug_config = config.get("debug", {})
    if debug_config.get("enabled", False):
        debug_mode = debug_config.get("mode", "quick_test")
        max_samples = debug_config.get("max_samples", 50)

        print(f"DEBUG MODE: {debug_mode}, limiting to {max_samples} samples")

        if debug_mode == "quick_test":
            # Quick test: take first N samples
            train_folds_df = train_folds_df.head(max_samples)
        elif debug_mode == "full_pipeline":
            # Full pipeline: stratified sample to ensure all classes present
            train_folds_df = train_folds_df.groupby("class_name").apply(
                lambda x: x.head(max(1, min(len(x), max_samples // 5)))
            ).reset_index(drop=True)
            # If we have fewer samples than requested, take more
            if len(train_folds_df) < max_samples:
                remaining = train_folds_df.head(max_samples)
                train_folds_df = remaining

    # Split by fold
    train_df = train_folds_df[train_folds_df["fold"] != fold].reset_index(drop=True)
    val_df = train_folds_df[train_folds_df["fold"] == fold].reset_index(drop=True)

    print(f"Fold {fold}: Train={len(train_df)}, Val={len(val_df)}")

    # Check for soundscape validation
    use_sc_val = config.get("validation", {}).get("use_soundscape_val", False)
    val_soundscape_loader = None

    if use_sc_val:
        # Load soundscape validation data
        sc_windows_path = Path(precomputed_dir) / "metadata" / "soundscape_windows.csv"
        if sc_windows_path.exists():
            sc_windows_df = pd.read_csv(sc_windows_path)
            sc_windows_df = sc_windows_df.drop_duplicates().reset_index(drop=True)

            if "split" not in sc_windows_df.columns:
                raise ValueError(
                    "soundscape_windows.csv is missing required 'split' column. "
                    "Regenerate precomputed metadata with the latest pipeline."
                )

            # Filter to validation split (held-out soundscape files)
            val_sc_df = sc_windows_df[sc_windows_df["split"] == "labeled_val"].reset_index(drop=True)

            if len(val_sc_df) > 0:
                print(f"Soundscape validation: {len(val_sc_df)} windows")

                # Apply debug limits
                if debug_config.get("enabled", False):
                    max_sc_windows = debug_config.get("max_sc_windows", 20)
                    val_sc_df = val_sc_df.head(max_sc_windows)
                    print(f"  Debug limited to: {len(val_sc_df)} windows")

                # Create soundscape validation dataset
                val_sc_dataset = SoundscapeDataset(
                    window_df=val_sc_df,
                    config=config,
                    mode="train",  # "train" mode includes labeled windows only
                    transform=None,
                )

                batch_size = config["training"]["batch_size"]
                num_workers = config["training"]["num_workers"]

                val_soundscape_loader = DataLoader(
                    val_sc_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=device_settings['pin_memory'],
                    drop_last=False,
                )

    # Build focal datasets
    use_perch = config.get("labels", {}).get("perch_alpha", 0.0) > 0
    perch_alpha = config.get("labels", {}).get("perch_alpha", 0.7)

    train_dataset = FocalDataset(
        meta_df=train_df,
        config=config,
        precomputed_dir=precomputed_dir,
        mode="train",
        use_perch_labels=use_perch,
        perch_alpha=perch_alpha,
        transform=None,  # Transform applied in training loop after augmentation
    )

    val_dataset = FocalDataset(
        meta_df=val_df,
        config=config,
        precomputed_dir=precomputed_dir,
        mode="val",
        use_perch_labels=False,  # Always use hard labels for validation
        transform=None,
    )

    # Build samplers
    # Training: WeightedRandomSampler based on precomputed sample weights
    train_weights = torch.FloatTensor([train_dataset.sample_weights[i] for i in range(len(train_dataset))])
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )

    # Build dataloaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device_settings['pin_memory'],
        drop_last=True,  # Drop last incomplete batch for consistent MixUp
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_settings['pin_memory'],
        drop_last=False,
    )

    # Store soundscape validation loader in config for later use
    if val_soundscape_loader is not None:
        config["_soundscape_val_loader"] = val_soundscape_loader

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    config: dict,
    device: torch.device,
    mel_transform: nn.Module,
    epoch: int,
) -> dict:
    """Train for one epoch.

    Args:
        model: Model to train
        loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        config: Config dict
        device: Torch device
        mel_transform: Mel spectrogram transform
        epoch: Current epoch number

    Returns:
        Dict with training metrics
    """
    model.train()
    mel_transform.train()

    # Get device-specific settings
    device_settings = get_device_settings(device)

    running_loss = 0.0
    num_batches = 0

    # Get augmentation config
    aug_config = config.get("augmentation", {})
    use_mixup = aug_config.get("use_mixup", True)
    mixup_alpha = aug_config.get("mixup_alpha", 0.5)
    mixup_prob = aug_config.get("mixup_prob", 0.5)

    grad_clip = config["training"].get("grad_clip", 1.0)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)

    for batch_idx, batch in enumerate(pbar):
        waveforms = batch["waveform"].to(device)  # [B, 160000]
        labels = batch["label"].to(device)  # [B, 234]

        # Apply audio augmentations (before mel conversion)
        # Random gain
        if aug_config.get("use_gain", True):
            gain_db = aug_config.get("gain_db", 6.0)
            for i in range(len(waveforms)):
                waveforms[i] = AudioAugmentation.random_gain(waveforms[i], gain_db=gain_db)

        # Random filtering
        if aug_config.get("use_random_eq", True):
            prob = aug_config.get("random_eq_prob", 0.5)
            sr = config["audio"]["sample_rate"]
            for i in range(len(waveforms)):
                waveforms[i] = AudioAugmentation.random_filtering(waveforms[i], sr=sr, prob=prob)

        # MixUp (audio-domain, not spectrogram-domain)
        if use_mixup and torch.rand(1).item() < mixup_prob:
            # Shuffle batch to create mixup pairs
            batch_size = len(waveforms)
            indices = torch.randperm(batch_size)

            mixed_waveforms = []
            mixed_labels = []

            for i in range(batch_size):
                j = indices[i].item()
                mixed_wave, mixed_label = AudioAugmentation.mixup(
                    waveforms[i],
                    labels[i],
                    waveforms[j],
                    labels[j],
                    alpha=mixup_alpha,
                )
                mixed_waveforms.append(mixed_wave)
                mixed_labels.append(mixed_label)

            waveforms = torch.stack(mixed_waveforms)
            labels = torch.stack(mixed_labels)

        # Convert to mel spectrogram
        with amp.autocast(device_settings['device_type']):
            mel = mel_transform(waveforms)  # [B, 1, 128, 313]
            mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_settings['device_type']):
            logits, probs = model(mel)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Track loss
        running_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{running_loss / num_batches:.4f}"})

    avg_loss = running_loss / max(num_batches, 1)

    return {"train_loss": avg_loss}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    config: dict,
    device: torch.device,
    mel_transform: nn.Module,
    taxonomy_df: pd.DataFrame,
    epoch: int,
) -> dict:
    """Validate model.

    Args:
        model: Model to validate
        loader: Validation dataloader
        criterion: Loss function
        config: Config dict
        device: Torch device
        mel_transform: Mel transform
        taxonomy_df: Taxonomy for per-class AUC
        epoch: Current epoch number

    Returns:
        Dict with validation metrics including per-class AUC
    """
    model.eval()
    mel_transform.eval()

    running_loss = 0.0
    num_batches = 0

    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False)

    for batch in pbar:
        waveforms = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        # Convert to mel (no augmentation)
        mel = mel_transform(waveforms)
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)

        # Forward pass
        logits, probs = model(mel)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            continue

        # Track loss and predictions
        running_loss += loss.item()
        num_batches += 1

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{running_loss / num_batches:.4f}"})

    # Concatenate all predictions
    all_probs = np.concatenate(all_probs, axis=0)  # [N, 234]
    all_labels = np.concatenate(all_labels, axis=0)  # [N, 234]
    all_probs = np.nan_to_num(all_probs, nan=0.0, posinf=1.0, neginf=0.0)
    all_labels = np.nan_to_num(all_labels, nan=0.0, posinf=1.0, neginf=0.0)

    # Compute metrics
    avg_loss = running_loss / max(num_batches, 1)

    # Overall macro AUC
    overall_auc = compute_macro_auc(all_probs, all_labels)

    # Per-class AUC breakdown
    per_class_auc = compute_per_class_auc(all_probs, all_labels, taxonomy_df)

    metrics = {
        "val_loss": avg_loss,
        "overall_auc": overall_auc,
        **{f"{cls}_auc": auc for cls, auc in per_class_auc.items() if cls != "overall"},
    }

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: dict,
    config: dict,
    is_best: bool = False,
):
    """Save model checkpoint.

    Handles checkpoint saving and optional Kaggle-compatible artifact export.
    Kaggle-compatible artifacts are exported only when kaggle mode is enabled.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch
        metrics: Validation metrics
        config: Config dict
        is_best: Whether this is the best checkpoint so far
    """
    checkpoint_dir = Path(config["paths"]["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Base checkpoint with full training state
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config,
    }

    # Always save latest checkpoint
    last_path = checkpoint_dir / "last_checkpoint.pth"
    torch.save(checkpoint, last_path)

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best_checkpoint.pth"
        torch.save(checkpoint, best_path)
        print(f"  Saved best checkpoint: {best_path}")

        # Export Kaggle-compatible model artifacts only for Kaggle runs
        kaggle_config = config.get("kaggle", {})
        export_kaggle_artifacts = kaggle_config.get("enabled", False)

        if export_kaggle_artifacts:
            print("  Saving Kaggle-compatible model artifacts...")

            # Save model state dict only (for Kaggle loading)
            kaggle_model_path = checkpoint_dir / "kaggle_model_state.pth"
            torch.save(model.state_dict(), kaggle_model_path)

            # Save model configuration for Kaggle reconstruction
            model_config = {
                "num_classes": config["project"]["num_classes"],
                "backbone": config["model"]["backbone"],
                "pretrained": config["model"]["pretrained"],
                "gem_p": config["model"]["gem_p"],
                "head_dropout": config["model"]["head_dropout"],
                "use_time_conditioning": False,  # Stage 2 doesn't use time
                "in_channels": 1,
            }

            kaggle_config_path = checkpoint_dir / "kaggle_model_config.json"
            import json
            with open(kaggle_config_path, "w") as f:
                json.dump(model_config, f, indent=2)

            # Save mel transform configuration
            audio_config = {
                "sample_rate": config["audio"]["sample_rate"],
                "n_fft": config["audio"]["n_fft"],
                "hop_length": config["audio"]["hop_length"],
                "n_mels": config["audio"]["n_mels"],
                "f_min": config["audio"]["f_min"],
                "f_max": config["audio"]["f_max"],
                "power": config["audio"]["power"],
                "use_pcen": True,
            }

            kaggle_audio_path = checkpoint_dir / "kaggle_audio_config.json"
            with open(kaggle_audio_path, "w") as f:
                json.dump(audio_config, f, indent=2)

            # Save final metrics for reference
            kaggle_metrics_path = checkpoint_dir / "kaggle_final_metrics.json"
            with open(kaggle_metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"    Model state: {kaggle_model_path}")
            print(f"    Model config: {kaggle_config_path}")
            print(f"    Audio config: {kaggle_audio_path}")
            print(f"    Final metrics: {kaggle_metrics_path}")
            print("  ✅ Kaggle-compatible artifacts ready for upload!")

            # Create a simple loading script for Kaggle
            loading_script = '''
# Kaggle Loading Script
# Copy this into your Kaggle notebook to load the trained model

import torch
import json
from src.model import build_model
from src.transforms import build_mel_transform

# Load configurations
with open("/kaggle/input/your-model-dataset/kaggle_model_config.json", "r") as f:
    model_config = json.load(f)

with open("/kaggle/input/your-model-dataset/kaggle_audio_config.json", "r") as f:
    audio_config = json.load(f)

# Rebuild model
model = build_model(**model_config)

# Load trained weights
model_state = torch.load("/kaggle/input/your-model-dataset/kaggle_model_state.pth")
model.load_state_dict(model_state)
model.eval()

# Rebuild audio transform
config = {"audio": audio_config}
mel_transform = build_mel_transform(config)

print("✅ Model loaded successfully!")
print(f"Model: {model_config['backbone']} with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
'''

            script_path = checkpoint_dir / "kaggle_loading_script.py"
            with open(script_path, "w") as f:
                f.write(loading_script)

            print(f"    Loading script: {script_path}")


def log_experiment(
    config: dict,
    fold: int,
    epoch: int,
    metrics: dict,
    notes: str = "",
):
    """Log experiment results to CSV.

    Uses configurable experiments path from config (supports both local and Kaggle paths).

    Args:
        config: Config dict
        fold: Fold number
        epoch: Epoch number
        metrics: Dict of metrics
        notes: Optional notes
    """
    # Use experiments path from config if available, otherwise default
    experiments_dir = Path(config.get("paths", {}).get("experiments", "./experiments"))
    log_path = experiments_dir / "log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Build row
    row = {
        "run_id": f"{Path(config.get('config_path', 'unknown')).stem}_fold{fold}_epoch{epoch}",
        "config": config.get("config_path", "unknown"),
        "fold": fold,
        "epoch": epoch,
        "overall_auc": metrics.get("overall_auc", 0.0),
        "aves_auc": metrics.get("Aves_auc", 0.0),
        "amphibia_auc": metrics.get("Amphibia_auc", 0.0),
        "insecta_auc": metrics.get("Insecta_auc", 0.0),
        "mammalia_auc": metrics.get("Mammalia_auc", 0.0),
        "reptilia_auc": metrics.get("Reptilia_auc", 0.0),
        "train_loss": metrics.get("train_loss", 0.0),
        "val_loss": metrics.get("val_loss", 0.0),
        "sc_val_auc": metrics.get("sc_val_auc", 0.0),  # Soundscape validation AUC
        "notes": notes,
    }

    # Write to CSV
    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train(config: dict, fold: int = 0):
    """Main training function.

    Args:
        config: Config dictionary
        fold: Fold number for cross-validation
    """
    # Set seed
    seed = config.get("project", {}).get("seed", 42)
    seed_everything(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load taxonomy
    taxonomy_path = config["paths"]["taxonomy"]
    taxonomy_df = pd.read_csv(taxonomy_path)

    # Build mel transform
    mel_transform = build_mel_transform(config).to(device)

    # Build dataloaders
    train_loader, val_loader = get_dataloaders(config, fold, mel_transform, device)

    # Build model
    model_config = config.get("model", {})
    model = build_model(
        num_classes=config["project"]["num_classes"],
        backbone=model_config.get("backbone", "efficientnet_b0"),
        pretrained=model_config.get("pretrained", True),
        gem_p=model_config.get("gem_p", 3.0),
        head_dropout=model_config.get("head_dropout", 0.25),
        use_time_conditioning=False,  # Not used in Stage 2
        in_channels=1,
    ).to(device)

    print(f"Model: {model_config.get('backbone', 'efficientnet_b0')}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Build loss
    criterion = get_loss(config)

    # Build optimizer
    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    # Build scheduler
    scheduler_type = train_config.get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_config["epochs"],
            eta_min=train_config.get("min_lr", 1e-6),
        )
    else:
        scheduler = None

    # Warmup scheduler (optional)
    warmup_epochs = train_config.get("warmup_epochs", 0)
    if warmup_epochs > 0:
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs),
        )

    # Get device-specific settings
    device_settings = get_device_settings(device)

    # Mixed precision scaler (only for CUDA)
    if device_settings['use_amp']:
        scaler = amp.GradScaler('cuda')
    else:
        scaler = amp.GradScaler('cpu')  # CPU scaler (essentially no-op)

    # Training loop
    best_auc = 0.0
    epochs = train_config["epochs"]

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            device=device,
            mel_transform=mel_transform,
            epoch=epoch,
        )
        # Report loss each epoch
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")

        # Print a brief summary every 5 epochs
        if epoch % 5 == 0:
            print(f"  Summary @ epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}")

        # Save checkpoint (training-only metrics)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=train_metrics,
            config=config,
            is_best=False,
        )

        # Log experiment (training-only)
        log_experiment(config, fold, epoch, train_metrics)

        # Step scheduler
        if epoch <= warmup_epochs and warmup_epochs > 0:
            warmup_scheduler.step()
        elif scheduler is not None:
            scheduler.step()

    # Final validation after training completes
    val_metrics = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        config=config,
        device=device,
        mel_transform=mel_transform,
        taxonomy_df=taxonomy_df,
        epoch=epochs,
    )

    # Optional soundscape validation
    sc_val_loader = config.get("_soundscape_val_loader")
    if sc_val_loader is not None:
        print(f"  Running soundscape validation...")
        sc_val_metrics = validate(
            model=model,
            loader=sc_val_loader,
            criterion=criterion,
            config=config,
            device=device,
            mel_transform=mel_transform,
            taxonomy_df=taxonomy_df,
            epoch=epochs,
        )

        # Weight soundscape validation in final score
        sc_weight = config.get("validation", {}).get("soundscape_weight", 0.3)
        focal_weight = 1.0 - sc_weight

        # Weighted combination of focal and soundscape validation
        combined_auc = (
            focal_weight * val_metrics["overall_auc"] +
            sc_weight * sc_val_metrics["overall_auc"]
        )

        val_metrics["sc_val_auc"] = sc_val_metrics["overall_auc"]
        val_metrics["combined_auc"] = combined_auc

        print(f"    Soundscape AUC: {sc_val_metrics['overall_auc']:.4f}")
        print(f"    Combined AUC: {combined_auc:.4f} (focal={focal_weight:.1f}, sc={sc_weight:.1f})")
    else:
        val_metrics["sc_val_auc"] = 0.0
        val_metrics["combined_auc"] = val_metrics["overall_auc"]

    # Combine and report final metrics
    all_metrics = {**val_metrics}
    best_auc = val_metrics["combined_auc"]

    print(f"\nFinal Validation Summary:")
    print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
    print(f"  Overall AUC: {val_metrics['overall_auc']:.4f}")
    print(f"  Aves AUC: {val_metrics.get('Aves_auc', 0.0):.4f}")
    print(f"  Amphibia AUC: {val_metrics.get('Amphibia_auc', 0.0):.4f}")
    print(f"  Insecta AUC: {val_metrics.get('Insecta_auc', 0.0):.4f}")
    print(f"  Mammalia AUC: {val_metrics.get('Mammalia_auc', 0.0):.4f}")

    # Save final checkpoint and log
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epochs,
        metrics=all_metrics,
        config=config,
        is_best=True,
    )

    log_experiment(config, fold, epochs, all_metrics, notes="final_validation")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train BirdCLEF 2026 model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number for cross-validation (default: 0)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config["config_path"] = args.config  # Store for logging

    # Run training
    train(config, fold=args.fold)


if __name__ == "__main__":
    main()
