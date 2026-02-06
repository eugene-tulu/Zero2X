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
        self.image_paths = []
        for d in image_dirs:
            for f in os.listdir(d):
                self.image_paths.append(os.path.join(d, f))

        self.image_paths.sort()
        self.label_dir = label_dir
        self.target_color = np.array(target_color)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        label_path = os.path.join(self.label_dir, filename)

        image = np.array(Image.open(img_path).convert("RGB"))
        label_rgb = np.array(Image.open(label_path).convert("RGB"))

        # Binary mask from RGB label
        mask = np.all(label_rgb == self.target_color, axis=-1).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask
