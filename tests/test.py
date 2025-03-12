import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.dataset_loader import load_images, encode_text, search_similar, transcribe_audio, image_filenames, index

device = "cuda" if torch.cuda.is_available() else "cpu"

load_images()

print("\nSelect input mode for search")
print("1. Text")
print("2. Voice")

while True:
    try:
        input_mode = int(input("\nEnter mode: ").strip())

        if input_mode == 1:
            query_text = input("\nEnter your text query: ").strip()
            break
        elif input_mode == 2:
            query_text = transcribe_audio()
            # print(f"\nTranscribed Query: {query_text}")
            break
        else:
            print("Invalid input! Please enter 1 or 2.")
    except ValueError:
        print("Invalid input! Please enter a number (1 or 2).")


query_embedding = encode_text([query_text])

# print("Query Embedding (first 5 values):", query_embedding[0][:5]) 

indices = search_similar(query_embedding, top_k=5)

print("\nQuery:", query_text)
print("Top matching images:")
for idx in indices[0]:
    if idx < len(image_filenames):
        print(f"Retrieved index: {idx}, Filename: {image_filenames[idx]}")
    else:
        print(f"Invalid index retrieved: {idx}")
