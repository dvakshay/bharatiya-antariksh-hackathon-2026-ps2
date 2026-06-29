import torch
from pathlib import Path

from PIL import Image
import torchvision.transforms as transforms

from config import *
from model import UNet

# unet architecture recreate
def load_model():

    model = UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODELS_DIR / "unet.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    return model

#Load One Cloudy Image
def load_image(image_path):

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return image, tensor

# predict cloud free images
def predict(model, image_tensor):

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        prediction = model(image_tensor)

    return prediction

#saving the cloud free image
def save_prediction(prediction, save_path):

    prediction = prediction.squeeze(0)

    prediction = prediction.cpu()

    prediction = transforms.ToPILImage()(prediction)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    prediction.save(save_path)

    print(f"Prediction saved to: {save_path}")

    #main function ensures sequence of excution
def main():
    print("Step 1: main() started")

    model = load_model()

    image_path = sorted(CLOUD_DIR.glob("*"))[0]

    print(f"Predicting : {image_path.name}")

    original_image, image_tensor = load_image(image_path)

    prediction = predict(model, image_tensor)

    save_prediction(
        prediction,
        RESULTS_DIR / "prediction.png"
    )


if __name__ == "__main__":
    main()