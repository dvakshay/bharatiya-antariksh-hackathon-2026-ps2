import csv
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import *
from model import UNet


# ==========================
# Load Model
# ==========================
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


transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])


# ==========================
# Predict
# ==========================
def predict(model, image):

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(tensor)

    prediction = prediction.squeeze(0).cpu()

    return prediction


# ==========================
# Main Evaluation
# ==========================
def evaluate():

    model = load_model()

    image_names = sorted(CLOUD_DIR.glob("*.png"))

    csv_path = RESULTS_DIR / "metrics.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_psnr = 0
    total_ssim = 0

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["Image", "PSNR", "SSIM"])

        for image_path in image_names:

            label_path = LABEL_DIR / image_path.name

            cloudy = Image.open(image_path).convert("RGB")

            ground_truth = Image.open(label_path).convert("RGB")

            prediction = predict(model, cloudy)

            pred = prediction.permute(1,2,0).numpy()

            gt = transform(ground_truth).permute(1,2,0).numpy()

            psnr = peak_signal_noise_ratio(gt, pred, data_range=1)

            ssim = structural_similarity(
                gt,
                pred,
                channel_axis=2,
                data_range=1
            )

            total_psnr += psnr
            total_ssim += ssim

            writer.writerow([
                image_path.name,
                round(psnr,4),
                round(ssim,4)
            ])

    avg_psnr = total_psnr / len(image_names)
    avg_ssim = total_ssim / len(image_names)

    print("="*50)
    print(f"Average PSNR : {avg_psnr:.4f}")
    print(f"Average SSIM : {avg_ssim:.4f}")
    print("="*50)

    print(f"\nMetrics saved to : {csv_path}")


if __name__ == "__main__":
    evaluate()