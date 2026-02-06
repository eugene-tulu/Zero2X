import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class GIDDataset(Dataset):
    def __init__(self, image_dirs, label_dir, target_color, transform=None):
        """
        image_dirs: list of folders containing images
        label_dir : folder containing all RGB label images
        target_color: (R, G, B) tuple
        """
        self.samples = []
        self.target_color = np.array(target_color)
        self.transform = transform

        label_files = set(os.listdir(label_dir))

        dropped = 0

        for d in image_dirs:
            for img_name in os.listdir(d):
                if not img_name.lower().endswith(".jpg"):
                    continue

                label_name = img_name.replace(".jpg", ".png")

                if label_name in label_files:
                    self.samples.append((
                        os.path.join(d, img_name),
                        os.path.join(label_dir, label_name)
                    ))
                else:
                    dropped += 1

        print(
            f"[GIDDataset] Loaded {len(self.samples)} image-label pairs "
            f"(dropped {dropped})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        label_rgb = np.array(Image.open(label_path).convert("RGB"))

        mask = np.all(label_rgb == self.target_color, axis=-1).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask