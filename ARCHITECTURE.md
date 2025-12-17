# RAG System Architecture

## Overview

This system implements a Retrieval-Augmented Generation (RAG) pipeline using:
- **Data Source**: JSON API Endpoint
- **Vector Database**: Pinecone
- **LLM**: Google Gemini
- **Embeddings**: sentence-transformers (local)

## Data Flow

```
JSON API Endpoint
     ↓
JSON Scraper → Text Chunker → Embedding Generator → Pinecone Vector Store
                                                          ↓
User Query → Embedding Generator → Pinecone Search → RAG ChatBot → Gemini → Response
```

## Components

### JSON Scraper
- Fetches data from configured JSON API endpoints
- Supports authentication and custom headers
- Handles pagination and batching

### Text Chunker
- Splits documents into overlapping chunks
- Preserves metadata across chunks
- Configurable chunk size and overlap

### Embedding Generator
- Uses sentence-transformers locally
- Default model: all-MiniLM-L6-v2 (384 dimensions)
- Batch processing support

### Pinecone Vector Store
- Cloud-hosted vector database
- REST API integration
- Supports metadata filtering
- Namespace isolation

### RAG ChatBot
- Google Gemini for generation
- Retrieval from Pinecone
- Conversation history support
- Source attribution

## Configuration

All configuration via environment variables:
- `PINECONE__API_KEY`: Pinecone API key
- `PINECONE__INDEX_URL`: Pinecone index URL
- `GEMINI__API_KEY`: Google Gemini API key
- `GEMINI__MODEL`: Model name (default: gemini-2.0-flash)