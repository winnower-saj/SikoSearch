import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.dataset_loader import load_images, encode_text, search_similar, image_filenames, index

device = "cuda" if torch.cuda.is_available() else "cpu"

load_images()

query_text = "Serene nature" # sample text for now

query_embedding = encode_text([query_text])

print("Query Embedding (first 5 values):", query_embedding[0][:5]) 

indices = search_similar(query_embedding, top_k=5)

print("\nQuery:", query_text)
print("Top matching images:")
for idx in indices[0]:
    if idx < len(image_filenames):
        print(f"Retrieved index: {idx}, Filename: {image_filenames[idx]}")
    else:
        print(f"Invalid index retrieved: {idx}")
