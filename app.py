import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
from dotenv import load_dotenv
import torch
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Load environment variables
load_dotenv()

# Load the CLIP model and processor from Hugging Face
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to check if an image is of a supported format (PNG, JPEG, etc.)
def is_supported_image_format(img_url):
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # Add other formats if needed
    return any(img_url.lower().endswith(ext) for ext in supported_formats)

# Function to classify if an image is a painting or drawing
def is_painting_or_drawing(image_content):
    image = Image.open(BytesIO(image_content))

    inputs = processor(
        text=["a painting", "a drawing", "a sketch", "a piece of art", "a realistic image"], 
        images=image, 
        return_tensors="pt", 
        padding=True
    )
    outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    painting_prob = probs[0][0].item()  # "a painting"
    drawing_prob = probs[0][1].item()  # "a drawing"
    sketch_prob = probs[0][2].item()   # "a sketch"
    art_prob = probs[0][3].item()      # "a piece of art"
    realistic_prob = probs[0][4].item()  # "a realistic image"

    return (painting_prob + drawing_prob + sketch_prob + art_prob) > realistic_prob

# Function to initialize Safari WebDriver
def init_webdriver():
    # Initialize Safari WebDriver (no need for a separate driver installation on macOS)
    driver = webdriver.Safari()  # Safari doesn't require options like Chrome
    return driver

# Function to scrape images dynamically with Selenium and Safari
def scrape_images_with_selenium(urls, folder="images"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    driver = init_webdriver()  # Use Safari WebDriver
    
    for url in urls:
        print(f"Scraping {url}...")
        try:
            driver.get(url)
            time.sleep(5)  # Let the page load completely

            soup = BeautifulSoup(driver.page_source, "html.parser")
            img_tags = soup.find_all("img")

            for img in img_tags:
                img_url = img.get("src")
                if img_url:
                    img_url = urljoin(url, img_url)

                    if not is_supported_image_format(img_url):
                        print(f"Skipped {img_url} (unsupported format)")
                        continue

                    img_name = os.path.basename(img_url)
                    try:
                        img_response = requests.get(img_url)
                        img_response.raise_for_status()

                        if is_painting_or_drawing(img_response.content):
                            img_path = os.path.join(folder, img_name)
                            with open(img_path, "wb") as f:
                                f.write(img_response.content)
                            print(f"Downloaded and saved {img_name} (painting/drawing)")
                        else:
                            print(f"Skipped {img_name} (not a painting or drawing)")
                    except Exception as e:
                        print(f"Could not download {img_url}. Reason: {e}")
        except Exception as e:
            print(f"Error while processing {url}: {e}")
        finally:
            print(f"Finished scraping {url}")

    driver.quit()

# Get URLs from the environment and split them into a list
urls = os.getenv("URLS").split(',')

# Run the scraper with Selenium and Safari
scrape_images_with_selenium(urls)
