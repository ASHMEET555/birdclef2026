"""Local testing entry point for BirdCLEF 2026 pipeline.

Quick smoke test to validate that all components work together.
Run this before pushing to Kaggle to catch errors early.

Usage:
    python main.py                    # Run full local test
    python main.py --quick            # Quick mode (fewer samples)
    python main.py --component loss   # Test specific component only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        from src import dataset, loss, metrics, model, temporal, train, transforms, pseudo_label
        print("✓ All core modules import successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_loss():
    """Test loss functions."""
    print("\n" + "=" * 60)
    print("Testing loss functions...")
    print("=" * 60)

    from src.loss import ASLLoss, LabelSmoothingCE, get_loss

    # Create dummy data
    batch_size, num_classes = 4, 234
    logits = torch.randn(batch_size, num_classes)
    targets = torch.zeros(batch_size, num_classes)
    targets[0, 10] = 1.0
    targets[1, 20] = 0.3  # Soft label

    # Test ASL
    asl = ASLLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    loss_asl = asl(logits, targets)
    print(f"ASL Loss: {loss_asl.item():.4f}")
    assert loss_asl.item() > 0, "ASL loss should be positive"

    # Test LabelSmoothingCE
    ce = LabelSmoothingCE(alpha=0.05, num_classes=num_classes)
    loss_ce = ce(logits, targets)
    print(f"Label Smoothing CE Loss: {loss_ce.item():.4f}")
    assert loss_ce.item() > 0, "CE loss should be positive"

    # Test factory
    config = {"loss": {"type": "asl", "gamma_neg": 4, "gamma_pos": 0, "clip": 0.05}}
    loss_fn = get_loss(config)
    print(f"Factory created: {type(loss_fn).__name__}")

    print("✓ All loss functions work correctly")
    return True


def test_metrics():
    """Test metrics computation."""
    print("\n" + "=" * 60)
    print("Testing metrics...")
    print("=" * 60)

    from src.metrics import compute_macro_auc, compute_per_class_auc
    import pandas as pd

    # Create dummy predictions and labels
    num_samples, num_classes = 100, 234
    np.random.seed(42)

    probs = np.random.rand(num_samples, num_classes).astype(np.float32)
    labels = np.zeros((num_samples, num_classes), dtype=np.float32)

    # Add some positive labels
    for i in range(num_samples):
        num_positives = np.random.randint(1, 5)
        positive_indices = np.random.choice(num_classes, size=num_positives, replace=False)
        labels[i, positive_indices] = 1.0

    # Test macro AUC
    macro_auc = compute_macro_auc(probs, labels)
    print(f"Macro AUC: {macro_auc:.4f}")
    assert 0.0 <= macro_auc <= 1.0, "AUC must be in [0, 1]"

    # Test per-class AUC
    taxonomy_df = pd.DataFrame({
        "primary_label": [f"species_{i}" for i in range(num_classes)],
        "class_name": (["Aves"] * 162 + ["Amphibia"] * 32 + ["Insecta"] * 28 + ["Mammalia"] * 8 + ["Reptilia"] * 4)[:num_classes]
    })

    per_class_auc = compute_per_class_auc(probs, labels, taxonomy_df)
    print(f"Per-class AUC:")
    for class_name, auc in per_class_auc.items():
        print(f"  {class_name}: {auc:.4f}")

    print("✓ All metrics work correctly")
    return True


def test_temporal():
    """Test temporal features."""
    print("\n" + "=" * 60)
    print("Testing temporal features...")
    print("=" * 60)

    from src.temporal import extract_hour_from_filename, cyclic_time_encoding, apply_temporal_prior

    # Test hour extraction
    test_file = "BC2026_Train_0001_S01_20240315_173045.ogg"
    hour = extract_hour_from_filename(test_file)
    print(f"Extracted hour from {test_file}: {hour:.4f}")
    assert 0.0 <= hour < 24.0, "Hour must be in [0, 24)"

    # Test cyclic encoding
    sin_val, cos_val = cyclic_time_encoding(hour)
    print(f"Cyclic encoding: sin={sin_val:.4f}, cos={cos_val:.4f}")
    magnitude = np.sqrt(sin_val**2 + cos_val**2)
    assert abs(magnitude - 1.0) < 1e-6, "Must be on unit circle"

    # Test temporal prior application
    num_classes = 234
    probs = np.random.rand(num_classes).astype(np.float32)
    prior_species = np.random.rand(num_classes, 24).astype(np.float32)
    prior_species = prior_species / prior_species.sum(axis=1, keepdims=True)

    adjusted = apply_temporal_prior(probs, hour=17.5, prior_species=prior_species)
    print(f"Temporal prior applied: {adjusted.shape}")
    assert adjusted.shape == probs.shape, "Shape mismatch"

    print("✓ All temporal features work correctly")
    return True


def test_transforms():
    """Test audio transforms."""
    print("\n" + "=" * 60)
    print("Testing transforms...")
    print("=" * 60)

    try:
        from src.transforms import MelPCENTransform, MelSpecConfig, AudioAugmentation

        # Create dummy waveform
        sr = 32000
        duration = 5.0
        n_samples = int(sr * duration)
        waveform = torch.randn(n_samples)

        # Test Mel+PCEN
        mel_config = MelSpecConfig(
            sample_rate=sr,
            n_mels=128,
            f_min=20.0,
            f_max=16000.0,
        )
        mel_transform = MelPCENTransform(mel_config)
        mel = mel_transform(waveform)
        print(f"Mel shape: {mel.shape} (expected [1, 1, 128, 313])")
        assert mel.shape == (1, 1, 128, 313), f"Unexpected mel shape: {mel.shape}"

        # Test MixUp
        waveform_a = torch.randn(n_samples)
        waveform_b = torch.randn(n_samples)
        label_a = torch.zeros(234)
        label_b = torch.zeros(234)
        label_a[10] = 1.0
        label_b[20] = 1.0

        mixed_wave, mixed_label = AudioAugmentation.mixup(waveform_a, label_a, waveform_b, label_b)
        print(f"MixUp: waveform shape {mixed_wave.shape}, label sum {mixed_label.sum().item()}")

        print("✓ All transforms work correctly")
        return True
    except ImportError as e:
        print(f"⚠ Transforms test skipped (missing dependency: {e})")
        return True  # Don't fail if torchaudio not installed


def test_model():
    """Test model architecture."""
    print("\n" + "=" * 60)
    print("Testing model...")
    print("=" * 60)

    try:
        from src.model import build_model

        # Build model (no pretrained for faster test)
        model = build_model(
            num_classes=234,
            pretrained=False,
            use_time_conditioning=False,
        )

        print(f"Model created: EfficientNet-B0 + GEM + Head")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

        # Test forward pass
        batch_size = 2
        mel = torch.randn(batch_size, 1, 128, 313)

        logits, probs = model(mel)
        print(f"Input shape: {mel.shape}")
        print(f"Output logits: {logits.shape}")
        print(f"Output probs: {probs.shape}, range [{probs.min():.3f}, {probs.max():.3f}]")

        assert logits.shape == (batch_size, 234), f"Unexpected logits shape: {logits.shape}"
        assert probs.shape == (batch_size, 234), f"Unexpected probs shape: {probs.shape}"
        assert (probs >= 0).all() and (probs <= 1).all(), "Probs not in [0, 1]"

        print("✓ Model works correctly")
        return True
    except ImportError as e:
        print(f"⚠ Model test skipped (missing dependency: {e})")
        return True


def test_config_loading():
    """Test config file loading."""
    print("\n" + "=" * 60)
    print("Testing config loading...")
    print("=" * 60)

    from src.train import load_config

    # Test base config
    config_path = "configs/base_config.yaml"
    if not Path(config_path).exists():
        print(f"✗ Config file not found: {config_path}")
        return False

    config = load_config(config_path)
    print(f"Base config loaded:")
    print(f"  Project: {config.get('project', {}).get('name', 'unknown')}")
    print(f"  Num classes: {config.get('project', {}).get('num_classes', 'unknown')}")
    print(f"  Sample rate: {config.get('audio', {}).get('sample_rate', 'unknown')}")
    print(f"  Backbone: {config.get('model', {}).get('backbone', 'unknown')}")
    print(f"  Loss: {config.get('loss', {}).get('type', 'unknown')}")

    # Test local_debug config with inheritance
    debug_config_path = "configs/local_debug.yaml"
    if Path(debug_config_path).exists():
        debug_config = load_config(debug_config_path)
        print(f"\nLocal debug config loaded:")
        print(f"  Debug enabled: {debug_config.get('debug', {}).get('enabled', False)}")
        print(f"  Debug mode: {debug_config.get('debug', {}).get('mode', 'unknown')}")
        print(f"  Max samples: {debug_config.get('debug', {}).get('max_samples', 'unknown')}")
        print(f"  Epochs: {debug_config.get('training', {}).get('epochs', 'unknown')}")
        print(f"  Batch size: {debug_config.get('training', {}).get('batch_size', 'unknown')}")

    # Test local_full config
    full_config_path = "configs/local_full.yaml"
    if Path(full_config_path).exists():
        full_config = load_config(full_config_path)
        print(f"\nLocal full config loaded:")
        print(f"  Debug enabled: {full_config.get('debug', {}).get('enabled', False)}")
        print(f"  Debug mode: {full_config.get('debug', {}).get('mode', 'unknown')}")
        print(f"  Max samples: {full_config.get('debug', {}).get('max_samples', 'unknown')}")
        print(f"  Save Kaggle compatible: {full_config.get('debug', {}).get('save_kaggle_compatible', False)}")
        print(f"  Use soundscape val: {full_config.get('validation', {}).get('use_soundscape_val', False)}")

    # Test kaggle config
    kaggle_config_path = "configs/kaggle_base.yaml"
    if Path(kaggle_config_path).exists():
        kaggle_config = load_config(kaggle_config_path)
        print(f"\nKaggle config loaded:")
        print(f"  Kaggle enabled: {kaggle_config.get('kaggle', {}).get('enabled', False)}")
        print(f"  Batch size: {kaggle_config.get('training', {}).get('batch_size', 'unknown')}")
        print(f"  Epochs: {kaggle_config.get('training', {}).get('epochs', 'unknown')}")
        print(f"  Checkpoints path: {kaggle_config.get('paths', {}).get('checkpoints', 'unknown')}")

    print("✓ All config loading works correctly")
    return True


def test_local_full_config():
    """Test local full training configuration and Kaggle compatibility."""
    print("\n" + "=" * 60)
    print("Testing local full training config...")
    print("=" * 60)

    # Check if we have precomputed data and config
    if not Path("./precomputed").exists():
        print("✗ Cannot test local full training without precomputed data")
        return False

    config_path = "configs/local_full.yaml"
    if not Path(config_path).exists():
        print(f"✗ Config not found: {config_path}")
        return False

    from src.train import load_config

    try:
        config = load_config(config_path)
        print(f"✓ Local full config loaded successfully")
        print(f"  Debug mode: {config.get('debug', {}).get('mode', 'unknown')}")
        print(f"  Max samples: {config.get('debug', {}).get('max_samples', 'unknown')}")
        print(f"  Kaggle compatible saving: {config.get('debug', {}).get('save_kaggle_compatible', False)}")
        print(f"  Soundscape validation: {config.get('validation', {}).get('use_soundscape_val', False)}")
        print(f"  Epochs: {config.get('training', {}).get('epochs', 'unknown')}")
        print(f"  Batch size: {config.get('training', {}).get('batch_size', 'unknown')}")

        # Check paths
        checkpoints_path = Path(config.get("paths", {}).get("checkpoints", "./checkpoints"))
        experiments_path = Path(config.get("paths", {}).get("experiments", "./experiments"))

        print(f"  Checkpoints path: {checkpoints_path}")
        print(f"  Experiments path: {experiments_path}")

        print("✓ Local full training config is ready")
        print("  Run: python src/train.py --config configs/local_full.yaml --fold 0")
        print("  This will:")
        print("    - Train on 2000 samples (stratified by class)")
        print("    - Use soundscape validation if available")
        print("    - Save Kaggle-compatible model artifacts")
        print("    - Train for 5 epochs with local GPU settings")

        return True

    except Exception as e:
        print(f"✗ Local full config test failed: {e}")
        return False


def test_precomputed_loading():
    """Test loading precomputed artifacts."""
    print("\n" + "=" * 60)
    print("Testing precomputed artifacts loading...")
    print("=" * 60)

    precomputed_dir = Path("./precomputed")

    if not precomputed_dir.exists():
        print(f"✗ Precomputed directory not found: {precomputed_dir}")
        print("  This is expected if you haven't run the precompute pipeline yet.")
        return False

    # Check for required files
    required_files = [
        "metadata/train_folds.csv",
        "metadata/sample_weights.npy",
        "label_vectors/train_labels.npy",
    ]

    all_exist = True
    for rel_path in required_files:
        full_path = precomputed_dir / rel_path
        if full_path.exists():
            print(f"✓ Found: {rel_path}")

            # Try to load and show shape
            if rel_path.endswith(".npy"):
                data = np.load(full_path)
                print(f"  Shape: {data.shape}, dtype: {data.dtype}")
            elif rel_path.endswith(".csv"):
                import pandas as pd
                df = pd.read_csv(full_path)
                print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        else:
            print(f"✗ Missing: {rel_path}")
            all_exist = False

    if all_exist:
        print("✓ All precomputed artifacts found")
    else:
        print("⚠ Some precomputed artifacts missing (run precompute pipeline)")

    return all_exist


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Local testing for BirdCLEF 2026")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (skip imports/dependencies check)"
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["imports", "loss", "metrics", "temporal", "transforms", "model", "config", "precomputed", "local_full"],
        help="Test specific component only",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("BirdCLEF 2026 Local Testing")
    print("="*60)

    results = {}

    # Define all tests
    tests = {
        "imports": test_imports,
        "loss": test_loss,
        "metrics": test_metrics,
        "temporal": test_temporal,
        "transforms": test_transforms,
        "model": test_model,
        "config": test_config_loading,
        "precomputed": test_precomputed_loading,
    }

    # Run selected tests
    if args.component:
        # Test specific component
        if args.component in tests:
            try:
                results[args.component] = tests[args.component]()
            except Exception as e:
                print(f"\n✗ {args.component} test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                results[args.component] = False
        elif args.component == "local_full":
            try:
                results["local_full"] = test_local_full_config()
            except Exception as e:
                print(f"\n✗ local_full test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                results["local_full"] = False
    else:
        # Run all core tests
        for name, test_fn in tests.items():
            try:
                results[name] = test_fn()
            except Exception as e:
                print(f"\n✗ {name} test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                results[name] = False

        # Run local_full test if not quick mode
        if not args.quick:
            try:
                results["local_full"] = test_local_full_config()
            except Exception as e:
                print(f"\n✗ local_full test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                results["local_full"] = False

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Ready for Kaggle submission.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Fix issues before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
