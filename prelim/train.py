import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import get_loaders, save_checkpoint, load_checkpoint, check_accuracy

# ==========================
# CONFIG
# ==========================
TRACK = "builtup"  # farmland | water | builtup

TRACK_COLORS = {
    "farmland": (0, 255, 0),
    "water": (0, 0, 255),
    "builtup": (255, 0, 0),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 5
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

TRAIN_IMAGE_DIRS = [
    "./data/1",
    "./data/2",
    "./data/3",
]

VAL_IMAGE_DIRS = [
    "./data/4",
]

LABEL_DIR = "./data/GID-label"


# ==========================
# LOSS FUNCTION (BCE + Dice)
# ==========================
class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)

        preds = torch.sigmoid(preds)
        smooth = 1e-6

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + smooth) / (
            preds.sum() + targets.sum() + smooth
        )
        dice_loss = 1 - dice

        return bce + dice_loss


# ==========================
# TRAIN FUNCTION
# ==========================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    running_loss = 0

    for data, targets in loop:
        data = data.to(DEVICE)
        targets = targets.unsqueeze(1).to(DEVICE)

        with torch.amp.autocast("cuda"):
            preds = model(data)
            loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)


# ==========================
# MAIN
# ==========================
def main():

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255),
        ToTensorV2(),
    ])

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    # Handle imbalance (built-up is sparse)
    pos_weight = torch.tensor([5.0]).to(DEVICE)

    loss_fn = BCEDiceLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMAGE_DIRS,
        LABEL_DIR,
        VAL_IMAGE_DIRS,
        LABEL_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        TRACK_COLORS[TRACK],
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        print(f"Train Loss: {train_loss:.4f}")

        check_accuracy(val_loader, model, DEVICE)

        save_checkpoint({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        })


if __name__ == "__main__":
    main()
