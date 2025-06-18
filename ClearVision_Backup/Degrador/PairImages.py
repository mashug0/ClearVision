from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class PairImages(Dataset):
    def __init__(self , clean_path , corrupted_path , transform = None):
        super().__init__()
        self.corrupted_path = corrupted_path
        self.clean_path = clean_path
        self.image_files = sorted(os.listdir(clean_path))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        filename = self.image_files[index]
        clean_image_path = os.path.join(self.clean_path , filename)
        corrupt_image_path = os.path.join(self.corrupted_path , filename)
        
        clean_img = Image.open(clean_image_path).convert("IMG")
        corrupt_img = Image.open(corrupt_image_path).convert("IMG")
        
        if self.transform:
            clean_img = self.transform(clean_img)
            corrupt_img = self.transform(corrupt_img)
        
        return corrupt_img , clean_img