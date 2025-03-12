# SikoSearch: Multi-Modal AI Search Engine (Prototype - Work in Progress)

SikoSearch is a multi-modal AI search engine that accepts various input types, including text, voice, and images, and returns diverse search results such as text, images, and more.

## Goal: A Fully Multi-Modal Search Engine
The aim is to develop a robust, AI-powered search engine capable of handling different modes of input and delivering various types of search results, similar to a general-purpose search engine.

The engine will eventually support:
- Text-to-Image (retrieve images from text queries)
- Speech-to-Image (retrieve images from spoken queries)
- Image-to-Image (find similar images based on an uploaded image)
- Video-to-Image (extract key frames from videos and retrieve similar images)
- Text-to-Text (retrieve relevant documents or web content)
- Hybrid Search (combine multiple input types for more refined results)

Currently, the prototype supports text- and voice-based image search.

## Features Implemented So Far
- Text-to-Image Search – Uses CLIP and FAISS to match text queries with relevant images.
- Real-Time Voice Search – Uses Whisper to transcribe spoken queries for image retrieval.
- FAISS Indexing – Provides fast similarity search for large datasets.
- Text-to-Text Search (Work In Progress) – Basic support for retrieving text-based results.
