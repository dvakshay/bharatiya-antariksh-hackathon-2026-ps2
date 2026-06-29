import torch
from PIL import Image
import torchvision.transforms as transforms

from config import *
from model import UNet


# =====================================
# Load Model Once
# =====================================
def load_model():

    model = UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODELS_DIR / "best_unet.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# Load only once
MODEL = load_model()


transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])


def predict_image(image):

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        prediction = MODEL(tensor)

    prediction = prediction.squeeze(0).cpu()

    prediction = transforms.ToPILImage()(prediction)

    return prediction