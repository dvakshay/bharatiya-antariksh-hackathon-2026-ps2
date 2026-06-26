from pathlib import Path
import torch

# ==========================
# Project Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "RICE"
RICE1_DIR = DATA_DIR / "RICE1"

CLOUD_DIR = RICE1_DIR / "cloud"
LABEL_DIR = RICE1_DIR / "label"

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

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
EPOCHS = 1

# ==========================
# Random Seed
# ==========================

SEED = 42