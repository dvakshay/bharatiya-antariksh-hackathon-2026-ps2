from pathlib import Path
import torch
import os

# ==========================
# Project Paths
# ==========================

# Detect if running in Google Colab
IS_COLAB = os.path.exists("/content")

if IS_COLAB:
    BASE_DIR = Path("/content/bharatiya-antariksh-hackathon-2026-ps2")

    DATA_DIR = Path("/content/drive/MyDrive/Bharatiya_Antariksh_Hackathon/RICE")
    RESULTS_DIR = Path("/content/drive/MyDrive/Bharatiya_Antariksh_Hackathon/results")
    MODELS_DIR = Path("/content/drive/MyDrive/Bharatiya_Antariksh_Hackathon/models")

else:
    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_DIR = BASE_DIR / "data" / "RICE"
    RESULTS_DIR = BASE_DIR / "results"
    MODELS_DIR = BASE_DIR / "models"

RICE1_DIR = DATA_DIR / "RICE1"

CLOUD_DIR = RICE1_DIR / "cloud"
LABEL_DIR = RICE1_DIR / "label"

# ==========================
# Image Settings
# ==========================

IMAGE_SIZE = (256, 256)

# ==========================
# Training Settings
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 20

# ==========================
# Random Seed
# ==========================

SEED = 42