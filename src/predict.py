import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from config import *
from model import UNet


# ==========================
# Load Trained Model
# ==========================
def load_model():

    model = UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
           import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from config import *
from model import UNet


# ==========================
# Load Trained Model
# ==========================
def load_model():

    model = UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
           import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from config import *
from model import UNet


# ==========================
# Load Trained Model
# ==========================
def load_model():

    model = UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
           MODELS_DIR / "unet.pth",,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# ==========================
# Load Image
# ==========================
def load_image(image_path):

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    return image, tensor


# ==========================
# Predict
# ==========================
def predict(model, tensor):

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        prediction = model(tensor)

    return prediction.squeeze(0).cpu()


# ==========================
# Save Tensor as Image
# ==========================
def save_tensor_image(tensor, path):

    image = transforms.ToPILImage()(tensor)

    image.save(path)


# ==========================
# Comparison Figure
# ==========================
def save_comparison(cloudy, prediction, ground_truth):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ax[0].imshow(cloudy)
    ax[0].set_title("Cloudy Input")
    ax[0].axis("off")

    ax[1].imshow(transforms.ToPILImage()(prediction))
    ax[1].set_title("Prediction")
    ax[1].axis("off")

    ax[2].imshow(ground_truth)
    ax[2].set_title("Ground Truth")
    ax[2].axis("off")

    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "comparison.png")

    plt.close()


# ==========================
# Main
# ==========================
def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    image_path = sorted(CLOUD_DIR.glob("*.png"))[0]

    label_path = LABEL_DIR / image_path.name

    cloudy_image, tensor = load_image(image_path)

    ground_truth = Image.open(label_path).convert("RGB")
    ground_truth = ground_truth.resize(IMAGE_SIZE)

    prediction = predict(model, tensor)

    save_tensor_image(
        prediction,
        RESULTS_DIR / "prediction.png"
    )

    cloudy_image.save(
        RESULTS_DIR / "cloudy.png"
    )

    ground_truth.save(
        RESULTS_DIR / "ground_truth.png"
    )

    save_comparison(
        cloudy_image,
        prediction,
        ground_truth
    )

    print("="*50)
    print("Prediction Completed Successfully!")
    print(f"Results saved in : {RESULTS_DIR}")
    print("="*50)


if __name__ == "__main__":
    main(),
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# ==========================
# Load Image
# ==========================
def load_image(image_path):

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    return image, tensor


# ==========================
# Predict
# ==========================
def predict(model, tensor):

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        prediction = model(tensor)

    return prediction.squeeze(0).cpu()


# ==========================
# Save Tensor as Image
# ==========================
def save_tensor_image(tensor, path):

    image = transforms.ToPILImage()(tensor)

    image.save(path)


# ==========================
# Comparison Figure
# ==========================
def save_comparison(cloudy, prediction, ground_truth):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ax[0].imshow(cloudy)
    ax[0].set_title("Cloudy Input")
    ax[0].axis("off")

    ax[1].imshow(transforms.ToPILImage()(prediction))
    ax[1].set_title("Prediction")
    ax[1].axis("off")

    ax[2].imshow(ground_truth)
    ax[2].set_title("Ground Truth")
    ax[2].axis("off")

    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "comparison.png")

    plt.close()


# ==========================
# Main
# ==========================
def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    image_path = sorted(CLOUD_DIR.glob("*.png"))[0]

    label_path = LABEL_DIR / image_path.name

    cloudy_image, tensor = load_image(image_path)

    ground_truth = Image.open(label_path).convert("RGB")
    ground_truth = ground_truth.resize(IMAGE_SIZE)

    prediction = predict(model, tensor)

    save_tensor_image(
        prediction,
        RESULTS_DIR / "prediction.png"
    )

    cloudy_image.save(
        RESULTS_DIR / "cloudy.png"
    )

    ground_truth.save(
        RESULTS_DIR / "ground_truth.png"
    )

    save_comparison(
        cloudy_image,
        prediction,
        ground_truth
    )

    print("="*50)
    print("Prediction Completed Successfully!")
    print(f"Results saved in : {RESULTS_DIR}")
    print("="*50)


if __name__ == "__main__":
    main(),
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# ==========================
# Load Image
# ==========================
def load_image(image_path):

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    return image, tensor


# ==========================
# Predict
# ==========================
def predict(model, tensor):

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        prediction = model(tensor)

    return prediction.squeeze(0).cpu()


# ==========================
# Save Tensor as Image
# ==========================
def save_tensor_image(tensor, path):

    image = transforms.ToPILImage()(tensor)

    image.save(path)


# ==========================
# Comparison Figure
# ==========================
def save_comparison(cloudy, prediction, ground_truth):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ax[0].imshow(cloudy)
    ax[0].set_title("Cloudy Input")
    ax[0].axis("off")

    ax[1].imshow(transforms.ToPILImage()(prediction))
    ax[1].set_title("Prediction")
    ax[1].axis("off")

    ax[2].imshow(ground_truth)
    ax[2].set_title("Ground Truth")
    ax[2].axis("off")

    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "comparison.png")

    plt.close()


# ==========================
# Main
# ==========================
def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    image_path = sorted(CLOUD_DIR.glob("*.png"))[0]

    label_path = LABEL_DIR / image_path.name

    cloudy_image, tensor = load_image(image_path)

    ground_truth = Image.open(label_path).convert("RGB")
    ground_truth = ground_truth.resize(IMAGE_SIZE)

    prediction = predict(model, tensor)

    save_tensor_image(
        prediction,
        RESULTS_DIR / "prediction.png"
    )

    cloudy_image.save(
        RESULTS_DIR / "cloudy.png"
    )

    ground_truth.save(
        RESULTS_DIR / "ground_truth.png"
    )

    save_comparison(
        cloudy_image,
        prediction,
        ground_truth
    )

    print("="*50)
    print("Prediction Completed Successfully!")
    print(f"Results saved in : {RESULTS_DIR}")
    print("="*50)


if __name__ == "__main__":
    main()