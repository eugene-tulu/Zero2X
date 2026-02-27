import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET

# ==========================
# CONFIG
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./my_checkpoint.pth.tar"
TEST_DIR = "./data/4"
OUTPUT_DIR = "./submission_masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# LOAD MODEL
# ==========================
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# ==========================
# TRANSFORM
# ==========================
transform = A.Compose([
    A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255),
    ToTensorV2(),
])

# ==========================
# INFERENCE LOOP
# ==========================
with torch.no_grad():
    for filename in os.listdir(TEST_DIR):
        if not filename.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(TEST_DIR, filename)
        image = np.array(Image.open(image_path).convert("RGB"))

        augmented = transform(image=image)
        image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

        preds = torch.sigmoid(model(image_tensor))
        preds = (preds > 0.5).float()

        mask = preds.squeeze().cpu().numpy().astype(np.uint8)

        # Ensure values are 0 or 1
        mask[mask > 0] = 1

        save_name = filename.replace(".jpg", ".png")
        save_path = os.path.join(OUTPUT_DIR, save_name)

        Image.fromarray(mask).save(save_path)

print("Submission masks generated successfully.")
