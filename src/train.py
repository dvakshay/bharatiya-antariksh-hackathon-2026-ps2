import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from config import *
from dataset import RiceCloudDataset
from model import UNet


def train():

    # Dataset
    dataset = RiceCloudDataset()

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Model
    model = UNet().to(DEVICE)

    # Loss Function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(f"\nTraining on: {DEVICE}")
    print(f"Total Images : {len(dataset)}")
    print(f"Total Batches: {len(train_loader)}\n")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )

        for cloud, label in progress_bar:

            cloud = cloud.to(DEVICE)
            label = label.to(DEVICE)

            prediction = model(cloud)

            loss = criterion(prediction, label)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Average Loss : {epoch_loss:.6f}")

    # Save Model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_path = MODELS_DIR / "unet.pth"

    torch.save(model.state_dict(), save_path)

    print("\n================================")
    print("Training Completed Successfully!")
    print(f"Model saved to : {save_path}")
    print("================================")


if __name__ == "__main__":
    train()