import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
from dotenv import load_dotenv
import torch

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

    # Expanded prompts for better classification
    inputs = processor(
        text=["a painting", "a drawing", "a sketch", "a piece of art", "a realistic image"], 
        images=image, 
        return_tensors="pt", 
        padding=True
    )
    outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Extract probabilities for each category
    painting_prob = probs[0][0].item()  # "a painting"
    drawing_prob = probs[0][1].item()  # "a drawing"
    sketch_prob = probs[0][2].item()   # "a sketch"
    art_prob = probs[0][3].item()      # "a piece of art"
    realistic_prob = probs[0][4].item()  # "a realistic image"

    # Return True if painting or drawing has a higher probability than realistic
    return (painting_prob + drawing_prob + sketch_prob + art_prob) > realistic_prob

# Function to scrape images from multiple URLs
def scrape_images(urls, folder="images"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for url in urls:
        print(f"Scraping {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            img_tags = soup.find_all("img")

            for img in img_tags:
                img_url = img.get("src")
                if img_url:
                    img_url = urljoin(url, img_url)
                    
                    # Skip if the file is not a supported image format (e.g., skip SVG files)
                    if not is_supported_image_format(img_url):
                        print(f"Skipped {img_url} (unsupported format)")
                        continue

                    img_name = os.path.basename(img_url)
                    try:
                        img_response = requests.get(img_url)
                        img_response.raise_for_status()

                        # Only save the image if it's classified as a painting or drawing
                        if is_painting_or_drawing(img_response.content):
                            img_path = os.path.join(folder, img_name)
                            with open(img_path, "wb") as f:
                                f.write(img_response.content)
                            print(f"Downloaded and saved {img_name} (painting/drawing)")
                        else:
                            print(f"Skipped {img_name} (not a painting or drawing)")
                    except Exception as e:
                        print(f"Could not download {img_url}. Reason: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve the webpage: {url}. Error: {e}")

# Get URLs from the environment and split them into a list
urls = os.getenv("URLS").split(',')

# Run the scraper with the list of URLs
scrape_images(urls)
