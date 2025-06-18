import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PairImages(Dataset):
    def __init__(self, clean_path, corrupted_path, transform=None):
        self.clean_path = clean_path
        self.corrupted_path = corrupted_path
        self.clean_images = sorted([os.path.join(clean_path, f) for f in os.listdir(clean_path) if f.endswith('.jpg') or f.endswith('.png')])
        self.degraded_images = sorted([os.path.join(corrupted_path, f) for f in os.listdir(corrupted_path) if f.endswith('.jpg') or f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.degraded_images)

    def __getitem__(self, idx):
        clean_img_path = self.clean_images[idx]
        degraded_img_path = self.degraded_images[idx]

        try:
            # Corrected conversion mode from "IMG" to "RGB"
            clean_image = Image.open(clean_img_path).convert("RGB")
            degraded_image = Image.open(degraded_img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return None or handle the error appropriately
            return None, None

        if self.transform:
            clean = self.transform(clean_image)
            degraded = self.transform(degraded_image)

        return degraded, clean