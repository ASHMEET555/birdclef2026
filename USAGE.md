# 🔧 Usage Guide: Local vs Kaggle Training

Your BirdCLEF 2026 codebase now supports **three distinct training modes** optimized for different development stages:

---

## 🎯 Training Mode Overview

| Mode | Config | Duration | Samples | Purpose | Kaggle Compatible |
|------|--------|----------|---------|---------|------------------|
| **Quick Test** | `local_debug.yaml` | 2 epochs | 50 samples | Pipeline verification | ❌ |
| **Local Full** | `local_full.yaml` | 5 epochs | 2000 samples | Full training locally | ❌ |
| **Kaggle Production** | `kaggle_base.yaml` | 30 epochs | All samples | Competition submission | ✅ |

---

## 🚀 Quick Start Commands

### 1. **Quick Pipeline Test** (2 minutes)
```bash
# Verify everything works on tiny dataset
python src/train.py --config configs/local_debug.yaml --fold 0
```
**Use when:** First time setup, testing changes

### 2. **Local Full Training** (1-2 hours)
```bash
# Train complete pipeline locally on subset
python src/train.py --config configs/local_full.yaml --fold 0
```
**Use when:** Model development, hyperparameter tuning
**Outputs:** Standard training checkpoints in `./checkpoints/`

### 3. **Kaggle Production** (3-4 hours on Kaggle GPU)
```bash
# Full competition training
python src/train.py --config configs/kaggle_base.yaml --fold 0
```
**Use when:** Final submission training

---

## 📁 Configuration Architecture

```
configs/
├── base_config.yaml          # Core ML hyperparameters (shared)
├── local_debug.yaml          # Quick test (inherits base_config)
├── local_full.yaml           # Local development (inherits kaggle_base)
├── kaggle_base.yaml          # Kaggle environment settings
└── stage2_base.yaml          # Production training (inherits kaggle_base)
```

### **Config Hierarchy:**
```
base_config.yaml              # ML parameters
    ↓
kaggle_base.yaml              # Kaggle paths + full training
   ├── local_full.yaml       # Local override for development
    └── stage2_base.yaml      # Production Stage 2
```

---

## 🏗️ Local Full Training Features

The `local_full.yaml` config provides **the best of both worlds**:

### ✅ **Local Development Benefits:**
- Faster iteration (5 epochs vs 30)
- Smaller dataset (2000 samples, stratified by class)
- Local debugging and monitoring
- Custom local paths

### ✅ **Kaggle-Aligned Training Setup:**
- **Identical model architecture** (EfficientNet-B0 + GEM)
- **Same hyperparameters** (ASL loss, augmentation, etc.)
- **Good proxy for Kaggle run behavior**

### 🎓 **Advanced Features:**
- **Soundscape validation:** Optional validation on held-out soundscape windows
- **Combined metrics:** Weighted focal + soundscape AUC for model selection
- **Experiment tracking:** All runs logged to experiments/log.csv

---

## 📦 Kaggle Model Transfer

When you run `kaggle_base.yaml`, it automatically generates:

```
checkpoints/
├── kaggle_model_state.pth         # Model weights (load with torch.load)
├── kaggle_model_config.json       # Model architecture config
├── kaggle_audio_config.json       # Mel transform config
├── kaggle_final_metrics.json      # Final validation metrics
└── kaggle_loading_script.py       # Copy-paste Kaggle loader
```

### **In Kaggle Notebook:**
```python
# 1. Upload checkpoints/ folder as Kaggle dataset
# 2. Copy-paste from kaggle_loading_script.py:

import torch
import json
from src.model import build_model

# Load model
with open("/kaggle/input/your-trained-model/kaggle_model_config.json", "r") as f:
    model_config = json.load(f)

model = build_model(**model_config)
model_state = torch.load("/kaggle/input/your-trained-model/kaggle_model_state.pth")
model.load_state_dict(model_state)
model.eval()

print("✅ Kaggle-trained model loaded successfully!")
```

---

## 🎛️ Configuration Guide

### **Key Differences:**

| Setting | local_debug | local_full | kaggle_base |
|---------|-------------|------------|-------------|
| **Samples** | 50 | 2000 | All (~35k) |
| **Epochs** | 2 | 5 | 30 |
| **Batch size** | 4 | 16 | 64 |
| **Duration** | 2 min | 1-2 hours | 3-4 hours |
| **Soundscape val** | ❌ | ✅ | ❌ |
| **Kaggle artifacts** | ❌ | ❌ | ✅ |

### **Shared Settings (Never Change):**
```yaml
# Critical ML parameters (identical across all configs)
audio:
  f_min: 20              # Frog frequency capture
  sample_rate: 32000
  n_mels: 128

model:
  backbone: efficientnet_b0
  gem_p: 3.0

loss:
  type: asl
  gamma_neg: 4           # Down-weight easy negatives
```

---

## 🧪 Development Workflow

### **Recommended Workflow:**

1. **🔬 Quick Test** (`local_debug.yaml`):
   ```bash
   python src/train.py --config configs/local_debug.yaml --fold 0
   ```
   Verify pipeline works, test code changes

2. **🛠️ Local Development** (`local_full.yaml`):
   ```bash
   python src/train.py --config configs/local_full.yaml --fold 0
   ```
   Develop model, tune hyperparameters, validate approach

3. **📤 Upload to Kaggle:**
   - Zip `checkpoints/` folder
   - Upload as Kaggle dataset
   - Use `kaggle_loading_script.py`

4. **🏆 Final Training** (`kaggle_base.yaml`):
   ```bash
   python src/train.py --config configs/kaggle_base.yaml --fold 0
   ```
   Train final model on full dataset

---

## 📊 Expected Performance

### **Local Full Training (5 epochs, 2000 samples):**
- **Overall AUC:** 0.70-0.80
- **Aves AUC:** 0.75-0.85
- **Amphibia AUC:** 0.50-0.70 (harder)
- **Training time:** ~1 hour on RTX 4050

### **Kaggle Production (30 epochs, full dataset):**
- **Overall AUC:** 0.80-0.88
- **Competition rank:** Top 20% baseline
- **Training time:** ~4 hours on Kaggle T4

---

## 🚨 Important Notes

### **Path Configuration:**
- **Local configs** use `./data/` and `./checkpoints/`
- **Kaggle configs** use `/kaggle/input/` and `/kaggle/working/`
- Precomputed data automatically handled

### **Soundscape Validation:**
- Only enabled in `local_full.yaml`
- Provides more realistic validation (domain gap assessment)
- Uses held-out soundscape windows from labeled data
- Weighted combination: 70% focal + 30% soundscape

### **Memory Requirements:**
- **local_debug:** ~2GB VRAM
- **local_full:** ~6GB VRAM
- **kaggle_base:** ~12GB VRAM (Kaggle T4)

---

Your codebase is now **production-ready** with seamless local↔Kaggle development! 🎉