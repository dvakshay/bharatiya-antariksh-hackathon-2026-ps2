from pathlib import Path

# ==========================
# Project Paths
# ==========================

# Path to the project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
DATA_DIR = BASE_DIR / "data" / "RICE"
RICE1_DIR = DATA_DIR / "RICE1"

CLOUD_DIR = RICE1_DIR / "cloud"
LABEL_DIR = RICE1_DIR / "label"

# Output folders
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# ==========================
# Image Settings
# ==========================

IMAGE_SIZE = (256, 256)

# ==========================
# Training Settings
# ==========================

BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 20

# ==========================
# Random Seed
# ==========================

SEED = 42