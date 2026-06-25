from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLOUD_DIR, LABEL_DIR, IMAGE_SIZE


class RiceCloudDataset(Dataset):

    def __init__(self):

        self.cloud_images = sorted(CLOUD_DIR.glob("*.png"))
        self.label_images = sorted(LABEL_DIR.glob("*.png"))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
        ])

    def __len__(self):
        return len(self.cloud_images)

    def __getitem__(self, idx):

        cloud = cv2.imread(str(self.cloud_images[idx]))
        cloud = cv2.cvtColor(cloud, cv2.COLOR_BGR2RGB)

        label = cv2.imread(str(self.label_images[idx]))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        cloud = self.transform(cloud)
        label = self.transform(label)

        return cloud, label