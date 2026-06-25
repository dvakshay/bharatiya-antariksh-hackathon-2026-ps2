import sys
from pathlib import Path

# Add src folder to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset import RiceCloudDataset

dataset = RiceCloudDataset()

print("=" * 40)
print(f"Total image pairs : {len(dataset)}")

cloud, label = dataset[0]

print(f"Cloud image shape : {cloud.shape}")
print(f"Label image shape : {label.shape}")
print("=" * 40)