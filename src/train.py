import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import RiceCloudDataset
from model import UNet


def train():

    # ==========================
    # Dataset
    # ==========================
    dataset = RiceCloudDataset()

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # ==========================
    # Model
    # ==========================
    model = UNet().to(DEVICE)

    # ==========================
    # Loss Function
    # ==========================
    criterion = nn.MSELoss()

    # ==========================
    # Optimizer
    # ==========================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Create models folder
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    print("=" * 60)
    print(f"Training on : {DEVICE}")
    print(f"Total Images : {len(dataset)}")
    print(f"Total Batches : {len(train_loader)}")
    print("=" * 60)

    # ==========================
    # Training Loop
    # ==========================
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

            progress_bar.set_postfix(loss=f"{loss.item():.5f}")

        epoch_loss = running_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Average Loss : {epoch_loss:.6f}")

        # ==========================================
        # Save checkpoint every epoch
        # ==========================================
        checkpoint_path = MODELS_DIR / f"epoch_{epoch+1}.pth"

        torch.save(model.state_dict(), checkpoint_path)

        print(f"Checkpoint Saved : {checkpoint_path}")

        # ==========================================
        # Save best model
        # ==========================================
        if epoch_loss < best_loss:

            best_loss = epoch_loss

            best_model_path = MODELS_DIR / "best_unet.pth"

            torch.save(model.state_dict(), best_model_path)

            print("Best Model Updated!")

    # ==========================================
    # Save final model
    # ==========================================
    final_model_path = MODELS_DIR / "unet_final.pth"

    torch.save(model.state_dict(), final_model_path)

    print("\n" + "=" * 60)
    print("Training Completed Successfully!")
    print(f"Final Model : {final_model_path}")
    print(f"Best Model  : {MODELS_DIR / 'best_unet.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    train()