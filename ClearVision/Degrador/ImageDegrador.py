#File Contains Multiple Functions Dealing with resizeing images which also introduce a bit a of corruption
#to equal size and corrupting them

#Degradation Based on https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Designing_a_Practical_Degradation_Model_for_Deep_Blind_Image_Super-Resolution_ICCV_2021_paper.pdf

#As the degradation pipeline in the paper suggests
# we Blurr(Gaussian Anisotropic + Isotropic) -> Downsample (Bilinear/ Bicubic) 
# -> Add Noise to the data (Gaussian Nose + JPEG Noise)
# Here we use a special technique in which we apply these operation or subset of operations in random
# order
import os
import io
import cv2
import math
import random
import threading

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image , to_tensor
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

class ImageDegrador:
    
    def __init__(self ,
                 downsample_factor_range=(2, 4),
                 target_size = (128 , 128),
                 base_path  = "/content/ClearVision/data/Scrapped", 
                 clean_path = "/content/ClearVision/data/Clean",
                 corruption_path = "/content/ClearVision/data/Corrupted"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel_sizes = list(range(3, 10, 2))  
        self.kernel_size = random.choice(self.kernel_sizes)
        self.scale_factor = random.randint(2 , 4)
        self.target_height = 128
        self.target_width = 128
        
        if self.scale_factor == 2:
            self.iso_sigma = (0.1, 2.4)
            self.aniso_sigma = (0.5, 6)
        else:  # scale_factor == 4
            self.iso_sigma = (0.1, 2.8)
            self.aniso_sigma = (0.5, 8)

        self.downsample_factor_range = downsample_factor_range
        self.target_size = target_size
        self.base_path = base_path
        self.corruption_path = corruption_path
        self.clean_path = clean_path
        
        self.file_info_list = []
        try:
                for self.foldername, self.subfolders, self.filenames in os.walk(self.base_path):
                    for filename in self.filenames:
                        filepath = os.path.join(self.foldername, filename)
                        file_size = os.path.getsize(filepath)
                        self.file_info_list.append({
                            'name': filename,
                            'path': filepath,
                            'size': file_size
                        })
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    def kernel_iso(self ,size , sigmaX , sigmaY):
        
        gauss_ax = torch.arange(size , dtype=torch.float32) - size//2
        gaussX = torch.exp( -0.5 * (gauss_ax /sigmaX) ** 2)
        gaussY = torch.exp( -0.5 * (gauss_ax /sigmaY) ** 2)
        
        gaussX /= gaussX.sum()
        gaussY /= gaussY.sum()
        
        Kernel2d = torch.outer(gaussY , gaussX)
        return Kernel2d / Kernel2d.sum()

    def kernel_aniso(self , size,sigmaX , sigmaY  , rot):
        gaus_ax = torch.arange(size , dtype=torch.float32) - size//2
        
        xx , yy = torch.meshgrid(gaus_ax , gaus_ax , indexing='xy')
        
        x_rot = xx * torch.cos(rot) + yy * torch.sin(rot)
        y_rot = -xx * torch.sin(rot) + yy* torch.cos(rot)
        
        kernel2d = torch.exp(-0.5 * ((x_rot / sigmaX) ** 2 + (y_rot / sigmaY) ** 2))
        kernel2d /= kernel2d.sum()

        return kernel2d

    def B_iso(self , image : torch.Tensor ) -> torch.Tensor:
        #Applies Isotropic Gaussian Blurr
        
        c,h,w = image.shape
        
        
        sigma_iso = random.uniform(*self.iso_sigma)
        iso_kernel = self.kernel_iso(self.kernel_size , sigma_iso , sigma_iso)

        padding = F.pad(image.unsqueeze(0) , (self.kernel_size // 2 , ) *4 , mode='reflect')
        
        iso_kernel = iso_kernel.expand(c , 1, self.kernel_size , self.kernel_size).to(image.device)
        padding = F.conv2d(padding, iso_kernel, groups=c)
        print("completed ISO")
        return padding.squeeze(0)

    def B_aniso(self, image: torch.Tensor) -> torch.Tensor:
        c, h, w = image.shape
        kernel_size = min(self.kernel_size,  h , w)  # Make sure this is set before calling this function
        
        if kernel_size % 2 == 0:
            kernel_size -=1        
            

        sigma_anisoX = random.uniform(*self.aniso_sigma)
        sigma_anisoY = random.uniform(*self.aniso_sigma)
        rot = random.uniform(0, math.pi)

        kernel = self.kernel_aniso(kernel_size, sigma_anisoX, sigma_anisoY, torch.tensor(rot))
        print(f"Input shape: {image.shape}, Kernel shape: {kernel.shape}")

        kernel = kernel.to(dtype=torch.float32, device=image.device)  
        kernel = kernel.expand(c, 1, kernel_size, kernel_size)        

        image = image.to(dtype=torch.float32, device=image.device)
        image = image.unsqueeze(0)  
        padding = kernel_size // 2
        image = F.pad(image, (padding, padding, padding, padding), mode='reflect')

        out = F.conv2d(image, kernel, groups=c)
        print(f"Completed ANISO")

        return out.squeeze(0) 

    
    def N_guass(self , image: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(1/255 , 3/255)
        c,h,w = image.shape
        image = transforms.ToTensor()(image)
        noise_type = random.choices(['gen' , 'channel' , 'gs'] , weights = [0.2 , 0.4 , 0.4])[0]
        noise = torch.empty((c, h, w) , device=self.device)
        if noise_type == 'general':
            noise = torch.randn((c, h, w), device=self.device) * sigma
        elif noise_type == 'channel':
            noise = torch.randn((c, h, w), device=self.device) * sigma
        elif noise_type == 'grayscale':
            gray_noise = torch.randn((1, h, w), device=self.device) * sigma
            noise = gray_noise.repeat(c, 1, 1)
        noisy_guass_image = torch.clamp(image + noise , 0.0 , 1.0)
        print("completed GUASS BLURR")
        return noisy_guass_image

    
    def N_jpeg(self , image :torch.Tensor) -> torch.Tensor:
        quality = random.randint(30 ,95)
        
        pil_image = to_pil_image(image.clamp(0,1))
        
        buffer = io.BytesIO()
        pil_image.save(buffer , format ='JPEG' , quality=quality)
        buffer.seek(0)
        
        jpeg_img = Image.open(buffer)
        jpeg_tensor = to_tensor(jpeg_img.convert('RGB'))
        print("completed JPEG COMPRESSION")
        return jpeg_tensor
    
    # These Two Resize Operations won't be used because they are not as efficient But these are some
    # of my interpretation of resizing
    def resize_images(self):
        self.target_width = 128
        self.target_height = 128
        self.resized_images = []
        for filename ,filepath , filesize in self.file_info_list:
            image = cv2.imread(filepath)
            resized_image = cv2.resize(image , (self.img_width , self.img_height) ,interpolation=cv2.INTER_AREA)
            self.resized_images.append(resized_image)
            
    def Ds(self , image : Image) -> torch.Tensor:
            self.target_width = 128
            self.target_height = 128
            
            image_t = to_tensor(image)
            
            c,h,w = image_t.shape
            size_transform = transforms.Compose([
                transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Pad(padding=[0, 0, max(self.target_width - w , 0), max(self.target_height - h , 0)], fill=0),
                transforms.ToTensor()
            ])
            
            resized_tensor = size_transform(image)
            return resized_tensor
        
    def degrade(self , img : torch.Tensor) -> torch.Tensor:
        ops = [self.B_iso , self.B_aniso , self.N_guass , self.N_jpeg]
        
        op_selected = random.sample(ops , k= random.randint(2 , len(ops)))
        random.shuffle(op_selected)
        print("Applied degradations:", [op.__name__ for op in op_selected])
        
        for op in op_selected:
            print(f"Applying {op.__name__}")
            if(op.__name__ == "B_aniso"):
                try:
                    img = self.B_aniso(image=img)
                except Exception as e:
                    print(f"B_aniso failed: {e}. Using B_iso instead.")
                    img = self.B_iso(image=img)    
            else:
                img = op(img)
            if img is None:
                raise ValueError(f"{op.__name__} returned None")
        print(img.shape)
        return img
    
    def save_degraded_images(self , max_workers = os.cpu_count()):
        
        os.makedirs(self.corruption_path , exist_ok=True)
        lock = threading.Lock()
        
        def process_and_save(fileinfo):
            try:
                filepath = fileinfo['path']
                filename = fileinfo['name']
                print(f"reading {filename}")
                
                try:
                    image = Image.open(filepath).convert('RGB')
                except Exception as e:
                    print(f"Could not open image {filepath}: {e}")
                    return
                
                image_t = to_tensor(image).to(self.device)
                print(image_t.shape)
                degraded = self.degrade(image_t)
                degraded = degraded.to(dtype=torch.float32, device=self.device)
                print(f"Degarde Shape:{degraded.shape}")
                img_p = to_pil_image(degraded.cpu().clamp(0,1))
                
                save_path = os.path.join(self.corruption_path , filename)
                img_p.save(save_path , format = 'JPEG' , quality = 95)
                
                return f"Done: {filename}"
    
            except Exception as e:
                return f"Error on {fileinfo['name']}: {e}" 
        
        clean_info_list = []
        try:
                for foldername, subfolders, filenames in os.walk(self.clean_path):
                    for filename in filenames:
                        filepath = os.path.join(foldername, filename)
                        file_size = os.path.getsize(filepath)
                        clean_info_list.append({
                            'name': filename,
                            'path': filepath,
                            'size': file_size
                        })
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_and_save, fi) for fi in clean_info_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Parallel Processing", unit="img"):
            _ = f.result()  # Could be used for logging if needed
   
    
    def save_clean_images(self , max_workers = os.cpu_count()):
        
        os.makedirs(self.clean_path , exist_ok=True)
        lock = threading.Lock()
        
        def process_and_save(fileinfo):
            try:
                filepath = fileinfo['path']
                filename = fileinfo['name']
                print(f"reading {filename}")
                
                try:
                    image = Image.open(filepath).convert('RGB')
                except Exception as e:
                    print(f"Could not open image {filepath}: {e}")
                    return
                
                image_t = to_tensor(image).to(self.device)
                print(f"image size {image_t.shape}")
                resized = self.Ds(image= image)
                print(f"Resized Shape:{resized.shape}")
                
                img_p = to_pil_image(resized.cpu().clamp(0,1))
                
                save_path = os.path.join(self.clean_path , filename)
                img_p.save(save_path , format = 'JPEG' , quality = 95)
                
                return f"Done: {filename}"
            except Exception as e:
                return f"Error on {fileinfo['name']}: {e}" 
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_and_save, fi) for fi in self.file_info_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Parallel Processing", unit="img"):
            _ = f.result()  # Could be used for logging if needed