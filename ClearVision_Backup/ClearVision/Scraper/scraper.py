import os
import requests
import base64
import time
from selenium import webdriver
from selenium.webdriver.common.by import By

class Scraper:
    def __init__(self, data_path, search_term, num_images=10):
        self.data_path = data_path
        self.search_term = search_term
        self.num_images = num_images
        self.driver = None

    def create_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=options)

    def scroll_down(self, target_count = 100 , scroll_pause_time=2, max_scrolls=8):
        seen_urls = set()
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        for _ in range(max_scrolls):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

            # Break early if enough images are loaded
            images = self.driver.find_elements(By.TAG_NAME, 'img')
            for img in images:
                src = img.get_attribute('src') or img.get_attribute('data-src')
                if src:
                    seen_urls.add(src)
            if len(seen_urls) >= target_count:
                break

    def scrape_all_images(self):
        try:
            images = self.driver.find_elements(By.TAG_NAME, 'img')
            image_urls = []
            for img in images:
                image_url = img.get_attribute('src') or img.get_attribute('data-src')
                if image_url and "data:image/gif" not in image_url:
                    width = int(img.get_attribute('width') or 0)
                    height = int(img.get_attribute('height') or 0)
                    if width >= 128 and height >= 128:
                        image_urls.append(image_url)
            return image_urls
        except Exception as e:
            print(f"Error scraping images: {e}")
            return []

    def save_image(self, image_url, file_name, retry_count=3):
        try:
            file_path = os.path.join(self.data_path, f"{file_name}.jpg")

            if image_url.startswith('data:image/'):
                header, encoded = image_url.split(',', 1)
                image_data = base64.b64decode(encoded)
                with open(file_path, 'wb') as f:
                    f.write(image_data)
            else:
                for attempt in range(retry_count):
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        break
                    else:
                        print(f"Failed attempt {attempt+1} for image: {image_url}")
                        time.sleep(2)
        except Exception as e:
            print(f"Error saving image {file_name}: {e}")

    def scrape_and_save_images(self):
        os.makedirs(self.data_path , exist_ok=True)

        self.create_driver()
        self.driver.get(f"https://unsplash.com/s/photos/{self.search_term}")
        time.sleep(5)
        self.scroll_down()
        image_urls = self.scrape_all_images()
        image_urls = image_urls[:self.num_images]

        for index, image_url in enumerate(image_urls, start=1):
            file_name = f"{self.search_term.replace(' ', '_')}_{index}"
            self.save_image(image_url, file_name)

        print(f"Finished scraping {len(image_urls)} images for '{self.search_term}'")
        self.driver.quit()