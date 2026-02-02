# Sanskrit Document Retrieval-Augmented Generation (RAG) System

A complete RAG system for processing and answering queries based on Sanskrit documents, designed to run entirely on CPU-based inference using llama.cpp.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)

## 🎯 Overview

This RAG system implements a complete pipeline for:
1. **Document Ingestion**: Loading Sanskrit documents from `.txt`, `.pdf`, and `.docx` formats
2. **Preprocessing**: Cleaning and chunking Sanskrit text
3. **Indexing**: Creating vector embeddings for efficient retrieval
4. **Retrieval**: Finding relevant document chunks based on queries
5. **Generation**: Generating coherent responses using CPU-based LLM inference

## ✨ Features

- **Multi-format Support**: Handles `.txt`, `.pdf`, and `.docx` files
- **Sanskrit Text Processing**: Optimized for Devanagari script and transliterated text
- **CPU-Only Inference**: Uses llama.cpp for efficient CPU-based LLM inference
- **Vector Retrieval**: Semantic search using multilingual sentence transformers
- **RESTful API**: FastAPI-based server with interactive documentation
- **Modular Architecture**: Clean separation between retriever and generator components

## 🏗️ System Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Retriever     │─────▶│  Vector Store│
│  (Embeddings)   │      │  (Embeddings)│
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Generator     │─────▶│  llama.cpp   │
│   (LLM)         │      │  (CPU)       │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│  Final Response │
└─────────────────┘
```

### Components

1. **Document Loader**: Extracts text from various file formats
2. **Preprocessor**: Cleans and chunks Sanskrit text
3. **Vector Retriever**: Creates embeddings and performs semantic search
4. **LLM Generator**: Generates responses using llama.cpp
5. **RAG Pipeline**: Orchestrates the complete flow

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- CPU with multiple cores (for efficient inference)

### Step 1: Clone or Navigate to Project

```bash
cd RAG_Sanskrit_<YourName>
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

**Option 1: Use the installation script (Recommended)**

```bash
# Windows
install_requirements.bat

# Linux/Mac
bash install_requirements.sh
```

**Option 2: Manual installation**

```bash
# First, upgrade pip and install build tools
python -m pip install --upgrade pip
pip install setuptools>=65.0.0 wheel>=0.38.0

# Then install all dependencies
pip install -r requirements.txt

# Finally, install llama-cpp-python (CPU version)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

**Note**: The installation script ensures setuptools is installed first, which prevents build errors.

## 🔧 Setup

### Step 1: Prepare Documents

Place your Sanskrit documents (`.txt`, `.pdf`, or `.docx`) in the `data/` directory.

### Step 2: Download LLM Model

You need to download a GGUF format model for llama.cpp. Recommended models:

- **Small (for testing)**: [TinyLlama-1.1B](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
- **Medium**: [Llama-2-7B-Chat](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- **Large**: [Mistral-7B-Instruct](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

Download a model and place it in a `models/` directory (create if needed).

Example:
```bash
mkdir models
# Download model to models/ directory
```

### Step 3: Configure

Edit `code/config.yaml` or set environment variables:

```bash
# Windows PowerShell
$env:DATA_DIR="../data"
$env:MODEL_PATH="models/llama-2-7b-chat.Q4_K_M.gguf"
$env:EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Linux/Mac
export DATA_DIR="../data"
export MODEL_PATH="models/llama-2-7b-chat.Q4_K_M.gguf"
export EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

## 🚀 Usage

### Start the Server

```bash
cd code
python main.py --host 0.0.0.0 --port 8000 --data-dir ../data --model-path ../models/your-model.gguf
```

Or use environment variables:

```bash
python main.py
```

The server will start on `http://localhost:8000`

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI or `http://localhost:8000/redoc` for ReDoc.

### Query via API

#### Using curl:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic discussed?",
    "use_rag": true,
    "top_k": 5
  }'
```

#### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "संस्कृत दस्तावेज़ में क्या चर्चा की गई है?",
        "use_rag": True,
        "top_k": 5
    }
)

result = response.json()
print(result["answer"])
```

#### Using the Web Interface:

1. Open `http://localhost:8000/docs`
2. Click on `/query` endpoint
3. Click "Try it out"
4. Enter your question and click "Execute"

## 📡 API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint. Returns:
```json
{
  "status": "healthy",
  "indexed": true,
  "model_loaded": true
}
```

### `POST /query`
Query the RAG system.

**Request Body:**
```json
{
  "question": "Your question here",
  "use_rag": true,
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Generated answer...",
  "context": "Retrieved context...",
  "query": "Your question",
  "retrieved_sources": [
    {
      "source": "document.txt",
      "chunk_index": 0,
      "score": 0.85,
      "text_preview": "..."
    }
  ]
}
```

### `POST /index`
Manually trigger document indexing.

**Query Parameters:**
- `force` (bool): Force reindexing even if embeddings exist

## ⚙️ Configuration

### Environment Variables

- `DATA_DIR`: Path to documents directory (default: `../data`)
- `MODEL_PATH`: Path to GGUF model file
- `EMBEDDING_MODEL`: Sentence transformer model name

### Config File

Edit `code/config.yaml` for detailed configuration:

```yaml
data_dir: "../data"
model:
  path: "models/llama-2-7b-chat.Q4_K_M.gguf"
  n_ctx: 2048
  n_threads: 4
embedding:
  model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
chunking:
  chunk_size: 500
  chunk_overlap: 100
retrieval:
  top_k: 5
```

## 📁 Project Structure

```
RAG_Sanskrit_<YourName>/
├── code/
│   ├── main.py                 # FastAPI server
│   ├── rag_pipeline.py         # Main RAG pipeline
│   ├── document_loader.py      # Document loading and preprocessing
│   ├── retriever.py            # Vector-based retriever
│   ├── generator.py            # LLM generator (llama.cpp)
│   ├── requirements.txt        # Python dependencies
│   └── config.yaml             # Configuration file
├── data/
│   ├── *.txt                   # Sanskrit text documents
│   ├── *.pdf                   # Sanskrit PDF documents
│   ├── *.docx                  # Sanskrit DOCX documents
│   └── vector_store.npz        # Saved embeddings (auto-generated)
├── models/                     # GGUF model files (create if needed)
│   └── *.gguf
├── report/                     # Technical report (PDF)
└── README.md                   # This file
```

## 🔬 Technical Details

### Document Processing

- **Text Cleaning**: Removes excessive whitespace while preserving Sanskrit Unicode
- **Chunking**: Sentence-aware chunking with configurable overlap
- **Encoding**: UTF-8 with support for Devanagari script (U+0900-U+097F)

### Embeddings

- **Model**: `paraphrase-multilingual-MiniLM-L12-v2` (supports 50+ languages including Sanskrit)
- **Storage**: NumPy compressed format (`.npz`)
- **Similarity**: Cosine similarity for retrieval

### LLM Inference

- **Framework**: llama.cpp via `llama-cpp-python`
- **Format**: GGUF (GPT-Generated Unified Format)
- **Optimization**: CPU-optimized with multi-threading
- **Context Window**: Configurable (default: 2048 tokens)

### Performance Considerations

- **First Run**: Slower due to model downloads and embedding creation
- **Subsequent Runs**: Faster with cached embeddings
- **Memory**: ~2-4GB RAM for small models, 8GB+ for 7B models
- **CPU**: Multi-threading enabled for faster inference

## 🐛 Troubleshooting

### Model Not Found
- Ensure the model path is correct
- Download a GGUF format model
- Check file permissions

### Import Errors
- Activate virtual environment
- Reinstall requirements: `pip install -r requirements.txt`
- For llama-cpp-python, try CPU-only installation

### Out of Memory
- Use a smaller model (Q4_K_M or Q4_0 quantization)
- Reduce `n_ctx` in config
- Close other applications

### Documents Not Loading
- Check file format (`.txt`, `.pdf`, `.docx`)
- Verify file encoding (UTF-8 recommended)
- Check file permissions

## 📝 Notes

- The system automatically creates embeddings on first run
- Embeddings are cached in `data/vector_store.npz`
- Sanskrit text can be input in Devanagari or transliterated form
- The multilingual embedding model handles both formats

## 📄 License

This project is for educational/assignment purposes.

## 👤 Author

RAG_Sanskrit_<YourName>

---

For questions or issues, please refer to the technical report in the `report/` directory.
