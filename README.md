# BirdCLEF 2026 | Competition + Research Paper

**Owner:** Gaurav Upreti | IIIT Una
**Competition:** BirdCLEF 2026 Kaggle Competition
**Status:** ✅ Ready for Training

---

## 🎯 Quick Start

### 1. **Pipeline Verification** (2 minutes)
```bash
# Test all components work
python main.py

# Quick pipeline test (50 samples, 2 epochs)
python src/train.py --config configs/local_debug.yaml --fold 0
```

### 2. **Local Full Training** (1-2 hours) 🆕
```bash
# Complete training locally on 2000 samples + Kaggle-compatible artifacts
python src/train.py --config configs/local_full.yaml --fold 0
```
**✨ NEW:** Produces Kaggle-ready model weights for seamless transfer!

### 3. **Kaggle Production** (3-4 hours)
```bash
# Full dataset training on Kaggle
python src/train.py --config configs/kaggle_base.yaml --fold 0
```

📖 **Detailed Usage Guide:** See [USAGE.md](USAGE.md) for complete local↔Kaggle workflow

---

## 📊 Expected Results (Sanity Check)

After 1 epoch with EfficientNet-B0 + ASL loss:
- **Train Loss:** 0.05 – 0.15 (ASL scale differs from BCE)
- **Overall AUC:** 0.75 – 0.85
- **Aves AUC:** 0.80 – 0.88 (birds are easier)
- **Amphibia AUC:** 0.55 – 0.72 (frogs are harder)
- **Insecta AUC:** 0.50 – 0.65 (many zero-shot)

⚠️ **If Amphibia AUC < 0.50:** Check f_min=20Hz (not 50Hz) for frog frequencies.

---

## 🏗️ Architecture Overview

### Model: **EfficientNet-B0 + GEM + 2-Layer Head**
```
Input: [B, 1, 128, 313] mel spectrogram (f_min=20Hz, PCEN normalized)
↓
EfficientNet-B0 backbone (1→3 channel conversion, ImageNet pretrained)
↓
GEM Pooling (p=3.0, learnable) → [B, 1280]
↓
Linear(1280→512) → ReLU → Dropout(0.25) → Linear(512→234)
↓
Output: Sigmoid probabilities [B, 234]
```

### Loss: **Asymmetric Loss (ASL)**
- `gamma_neg=4` (strong down-weight easy negatives)
- `gamma_pos=0` (no modulation on positives)
- `clip=0.05` (decision boundary shift)
- **WHY:** 98% of labels are negative (absent species)

### Augmentation: **Audio-Domain (Confirmed +0.036 LB)**
- **MixUp:** Audio mixing before mel conversion
- **RandomGain:** ±6dB (bridges XC 10.3dB ↔ iNat 7.1dB SNR gap)
- **RandomFiltering:** Biquad EQ (microphone invariance)
- **SpecAugment:** Time/freq masking

---

## 📁 Key Files

| File | Purpose | Status |
|------|---------|---------|
| **`src/train.py`** | Main training orchestrator | ✅ Complete |
| **`src/loss.py`** | ASL, FocalLoss, LabelSmoothingCE | ✅ Complete |
| **`src/model.py`** | EfficientNet-B0 + GEM + Head | ✅ Complete |
| **`src/dataset.py`** | FocalDataset, SoundscapeDataset | ✅ Complete |
| **`src/transforms.py`** | Mel+PCEN, MixUp, SpecAugment | ✅ Complete |
| **`src/metrics.py`** | Macro AUC, per-class breakdown | ✅ Complete |
| **`src/temporal.py`** | Time encoding, activity priors | ✅ Complete |
| **`configs/base_config.yaml`** | All hyperparameters | ✅ Complete |
| **`main.py`** | Local testing suite | ✅ Complete |

---

## 🔧 Configuration

### **Config Hierarchy:**
```
base_config.yaml          # Core hyperparameters
├── local_debug.yaml      # Local development (50 samples, 2 epochs)
├── stage2_base.yaml      # Clean training (hard labels only)
├── stage3_pseudo.yaml    # + Pseudo-labels from Stage 2
└── stage4_finetune.yaml  # + Soundscape fine-tuning
```

### **Critical Settings (DO NOT CHANGE):**
```yaml
audio:
  f_min: 20         # NOT 50 - captures frog fundamentals
  sample_rate: 32000
  n_mels: 128

loss:
  type: asl         # Asymmetric Loss
  gamma_neg: 4      # Down-weight easy negatives

model:
  backbone: efficientnet_b0
  gem_p: 3.0        # GEM pooling power
```

---

## 🚀 Deploy to Kaggle

### **Method 1: Git Clone (Recommended)**
```python
!git clone https://github.com/gauravupreti/birdclef2026 /kaggle/working/birdclef2026
%cd /kaggle/working/birdclef2026
!python src/train.py --config configs/stage2_base.yaml --fold 0
```

### **Method 2: Upload as Dataset**
1. Zip entire project: `zip -r birdclef2026.zip . -x .git/\*`
2. Upload as Kaggle dataset
3. Add as data source to notebook

---

## 📈 Training Progression

1. **Stage 2:** Clean train_audio only (ASL loss, hard labels)
2. **Stage 3:** + Pseudo-labels from unlabeled soundscapes
3. **Stage 4:** + Direct soundscape fine-tuning
4. **Stage 5:** Model ensemble + temporal priors

**Current Status:** ✅ Stage 2 Ready

---

## 🧪 Experiments Tracking

All runs logged to `experiments/log.csv`:
```
run_id | config | fold | epoch | overall_auc | aves_auc | amphibia_auc | ...
```

**View results:**
```python
import pandas as pd
df = pd.read_csv("experiments/log.csv")
df.sort_values("overall_auc", ascending=False).head()
```

---

## 💾 Dependencies

**Core:** `torch` `torchaudio` `timm` `librosa` `pandas` `numpy` `scikit-learn`
**Optional:** `h5py` (Perch soft labels) `pyyaml` (configs) `tqdm` (progress bars)

**Install:**
```bash
pip install -r requirements.txt
# OR
uv sync  # If using uv (faster)
```

---

## 🏆 Competition Info

- **Task:** Multi-label audio classification (234 species)
- **Metric:** Macro ROC-AUC (equal weight per species)
- **Data:** Pantanal soundscapes (5-second windows)
- **Constraint:** CPU-only inference, 90-minute runtime
- **Deployment:** INT8 ONNX quantization required

---

## 🎓 Research Paper Contributions

1. **Temporal Activity Priors:** Sin/cos time encoding + species activity patterns
2. **Audio-Domain MixUp:** Waveform-level mixing (vs spectrogram-level)
3. **Tri-Taxonomic Analysis:** Birds vs Amphibians vs Insects performance breakdown
4. **Out-of-Fold Pseudo-Labeling:** Principled semi-supervised learning approach

---

**Ready to train!** 🚀 Run `python main.py` to verify everything works.