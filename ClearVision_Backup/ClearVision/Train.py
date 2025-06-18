from Scraper.scraper import Scraper
from Degrador.ImageDegrador import ImageDegrador
from Degrador.PairImages import PairImages
from torch.utils.data import DataLoader

import os

if __name__ == "__main__":
    
    data_path = r"ClearVision\\Data"
    
    search_terms = [
        # Nature & Landscapes
        "forest", "mountain", "river", "lake", "beach", "desert", "snow", "sunset", "sunrise", "sky", 
        "waterfall", "cliff", "valley", "field", "rainforest", "foggy forest", "countryside", "volcano", "iceberg", "stormy sky",
        
        # Urban & Architecture
        "city skyline", "urban street", "skyscraper", "bridge", "building facade", "night city", "abandoned building",
        "interior design", "subway station", "apartment", "old house", "modern house", "window light", "empty room", "hotel lobby",
        
        # People & Portraits
        "portrait", "man face", "woman face", "smiling child", "elderly person", "person walking", "person reading", 
        "silhouette", "group of people", "people in market", "candid street portrait", "close up face", "blurred person",
        
        # Daily Objects & Lifestyle
        "books", "laptop on desk", "coffee cup", "camera", "glasses", "shoes", "bicycle", "chair", "table", "clock",
        "phone", "headphones", "pen and notebook", "typewriter", "watch", "keys", "mirror", "umbrella", "shopping cart", "door handle",
        
        # Animals
        "cat", "dog", "horse", "bird", "owl", "elephant", "lion", "tiger", "monkey", "fish",
        "wild animal", "animal closeup", "farm animals", "animal eyes", "animal fur",
        
        # Food
        "fruits", "vegetables", "pizza", "burger", "pasta", "salad", "bread", "coffee", "tea", "ice cream",
        "kitchen interior", "table setting", "street food", "grocery store", "picnic food",
        
        # Transportation
        "car", "old car", "motorbike", "bicycle", "bus", "train", "airplane", "boat", "subway", "traffic jam",
        
        # Abstract, Texture & Patterns
        "fabric texture", "wood texture", "rusty metal", "peeling paint", "grungy wall", "glass reflection", 
        "water droplets", 
        "bokeh lights", "light rays", "scratched surface", "paper texture", "brick wall",
        "concrete surface", "cracked ground", "muddy surface",
        
        # Weather & Conditions
        "rainy day", "snowy street", "foggy morning", "storm clouds", "windy field", "misty lake", 
        "overcast sky", "bright sunlight", "night fog", "dusty road",

        # Miscellaneous / High Variety
        "vintage object", "minimalist interior", "industrial scene", "colorful buildings", "market street", 
        "library shelves", "playground", "school hallway", "concert crowd", "park bench"
    ]

    # for term in search_terms:
    #     scraper = Scraper(base_folder=data_path, search_term=term, num_images=50)
    #     scraper.scrape_and_save_images()

    degrador = ImageDegrador()
    # degrador.save_clean_images(max_workers= os.cpu_count())
    degrador.save_degraded_images(max_workers=os.cpu_count())
    
    # dataset = PairImages("ClearVision/Data/Corrupted" , "ClearVision/Data/Clean")
    # dataloader = DataLoader(dataset=dataset , batch_size= 32 , shuffle=True)