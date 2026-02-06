import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import GIDDataset

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_image_dirs,
    train_label_dir,
    val_image_dirs,
    val_label_dir,
    batch_size,
    train_transform,
    val_transform,
    target_color,
    num_workers=4,
    pin_memory=True,
):
    train_ds = GIDDataset(
        train_image_dirs,
        train_label_dir,
        target_color,
        train_transform,
    )

    val_ds = GIDDataset(
        val_image_dirs,
        val_label_dir,
        target_color,
        val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(f"Acc: {num_correct/num_pixels*100:.2f}%")
    print(f"Dice: {dice_score/len(loader):.4f}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = (torch.sigmoid(model(x)) > 0.5).float()

        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
    model.train()
