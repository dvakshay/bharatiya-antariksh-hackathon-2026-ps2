import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent / "src"))

from model import UNet

model = UNet()

x = torch.randn(1, 3, 256, 256)

y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)