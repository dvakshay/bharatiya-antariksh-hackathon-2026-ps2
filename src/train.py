#it is the dataloader
from torch.utils.data import DataLoader

from dataset import RiceCloudDataset
from config import BATCH_SIZE

# Create Dataset
dataset = RiceCloudDataset()

# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Dataset Size : {len(dataset)}")
print(f"Total Batches : {len(train_loader)}")

# Check one batch
cloud_batch, label_batch = next(iter(train_loader))

print(f"Cloud Batch Shape : {cloud_batch.shape}")
print(f"Label Batch Shape : {label_batch.shape}")