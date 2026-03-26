
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
