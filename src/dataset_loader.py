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

def transcribe_audio():
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1 
    rate = 16000 
    record_seconds = 5
    output_filename = "temp_audio.wav" 

    audio = pyaudio.PyAudio()

    print("\nRecording... Speak now!")
    stream = audio.open(format=format, channels=channels,
                        rate=rate, input=True,
                        frames_per_buffer=chunk)

    frames = []

    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(output_filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    waveform, sample_rate = sf.read(output_filename, dtype="float32")

    inputs = whisper_processor(waveform, sampling_rate=16000, return_tensors="pt", language="en").to(device)

    with torch.no_grad():
        predicted_ids = whisper_model.generate(inputs.input_features)

    transcription = whisper_processor.decode(predicted_ids[0])

    return transcription.strip()

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
