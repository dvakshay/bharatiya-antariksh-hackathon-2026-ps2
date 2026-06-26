import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import RiceCloudDataset
from model import UNet

# Dataset
dataset = RiceCloudDataset()

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Model
model = UNet().to(DEVICE)

# Loss
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("Training Started...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for cloud, label in tqdm(train_loader):

        cloud = cloud.to(DEVICE)
        label = label.to(DEVICE)

        prediction = model(cloud)

        loss = criterion(prediction, label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss : {epoch_loss:.6f}")