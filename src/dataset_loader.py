import torch
import os
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import faiss

from src.config import config 

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained(config["clip_model"]).to(device)
processor = CLIPProcessor.from_pretrained(config["clip_model"])

dim = 512 
index = faiss.IndexFlatL2(dim) 

data_dir = "data/sample_images/"

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
]) 

image_filenames = [] # global storage for now (for verifying)

def encode_images(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu().numpy().astype(np.float32)

def encode_text(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu().numpy().astype(np.float32)

def load_images():
    global image_filenames 
    image_embeddings = []

    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        image_embedding = encode_images(image)

        image_embeddings.append(image_embedding)
        image_filenames.append(img_name)

    image_embeddings = np.vstack(image_embeddings)

    print("Final shape of image embeddings:", image_embeddings.shape)
    print("Images indexed in FAISS:", image_filenames)

    index.add(image_embeddings)

    return image_filenames 
def search_similar(query_embedding, top_k=5):
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return indices
