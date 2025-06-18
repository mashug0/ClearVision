import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image , to_tensor
from PIL import Image
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy
from torchvision.utils import save_image
import random
import math
import io
from Degrador.ImageDegrador import ImageDegrador
import cv2
import time
device = torch.device("cuda" if torch.cuda.is_available()else "cpu")


# Step 1: Load image and convert to tensor
img_path = "D:\\Dev\\ClearVision\\Data\\Scrapped\\dog_13.jpg"  # <-- replace with your image path
image = Image.open(img_path).convert("RGB")
print(image.size)
# Resize (optional but recommended)
transform = transforms.ToTensor()
image_tensor = transform(image)  # shape: [3, 128, 128]
image_tensor = image_tensor.to(device= device)

# Step 2: Apply degradation
ax = torch.arange(7 , dtype = torch.float32) - 7//2
xx , yy = torch.meshgrid(ax , ax , indexing='xy')
rot = random.uniform(0 , math.pi )
rot = torch.tensor(rot)
x_rot = xx * torch.cos(rot) + yy * torch.sin(rot)
y_rot = -xx * torch.sin(rot) + yy* torch.cos(rot)
        
kernel2d = torch.exp(-0.5 * ((x_rot / 5) ** 2 + (y_rot / 6) ** 2))
kernel2d /= kernel2d.sum()
kernel2d = kernel2d.to(device=device)
print(image_tensor.shape)
c,h,w = image_tensor.shape
padding = F.pad(image_tensor.unsqueeze(0) , (7 // 2 , ) *4 , mode='reflect')
kernel = kernel2d.expand(c , 1 , 7 , 7).to(image_tensor.device)
padding = F.conv2d(padding, kernel, groups=c)

gauss = padding.squeeze(0)


# Step 3: Visualize original vs degraded
# def show_image(tensor_img, title=""):
#     np_img = tensor_img.permute(1, 2, 0).cpu().numpy()
#     plt.imshow(np_img)
#     plt.axis("off")
#     plt.title(title)
# def save_image(img_tensor : torch.Tensor , title =""):
#     np_img = img_tensor.permute(1,2,0).cpu().clamp(0,1).numpy()
#     plt.plot(np_img)
#     plt.title({title})
#     plt.savefig("D:\\Dev\\ClearVision\\{title}.png")

target_width = 128
target_height = 128

size_transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Pad(padding=[0, 0, max(target_width - w , 0), max(target_height - h , 0)], fill=0),
    transforms.ToTensor()
    
])
resize_image = size_transform(image)

sigma = 10/255
noise_type = random.choices(['general' , 'channel' ,'gs' ] , weights=[0.2 , 0.4 , 0.4])[0]
C, H, W = resize_image.shape

if noise_type == 'general':
    noise = torch.randn((C, H, W), device=device) * sigma
elif noise_type == 'channel':
    noise = torch.randn((C, H, W), device=device) * sigma
elif noise_type == 'grayscale':
    gray_noise = torch.randn((1, H, W), device=device) * sigma
    noise = gray_noise.repeat(C, 1, 1)
noised_image = torch.clamp(resize_image , 0.0 , 1.0)


pil_image = to_pil_image(noised_image)
buffer = io.BytesIO()
quality = random.randint(30 , 95)
pil_image.save(buffer , format ='JPEG' , quality = quality)
buffer.seek(0)

jpeg_img = Image.open(buffer)
jpeg_tensor = to_tensor(jpeg_img)


save_image(tensor=image_tensor ,fp= "Original.png")
# save_image(tensor=padding.squeeze(0) ,fp= "Blurred.png")
save_image(resize_image ,fp= "Resized.png")
# save_image(noised_image ,fp= "noised.png")
# save_image(jpeg_tensor ,fp= "JPEG.png")


# degrade = ImageDegrador(base_path="D:\\Dev\\ClearVision\\test" , corruption_path="D:\\Dev\\ClearVision\\corrupted")
# degrade.save_images(os.cpu_count())

