"""
FastAPI Server for Sanskrit RAG System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from rag_pipeline import RAGPipeline
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    use_rag: bool = True
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    context: Optional[str]
    query: str
    retrieved_sources: List[dict]


class HealthResponse(BaseModel):
    status: str
    indexed: bool
    model_loaded: bool


# Initialize FastAPI app
app = FastAPI(
    title="Sanskrit RAG System",
    description="Retrieval-Augmented Generation system for Sanskrit documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None


def initialize_pipeline():
    """Initialize the RAG pipeline"""
    global rag_pipeline
    
    # Get configuration from environment or use defaults
    data_dir = os.getenv("DATA_DIR", "../data")
    model_path = os.getenv("MODEL_PATH", None)
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k = int(os.getenv("TOP_K", "5"))
    
    print("Initializing RAG Pipeline...")
    print(f"  Data directory: {data_dir}")
    print(f"  Model path: {model_path if model_path else 'Not set'}")
    print(f"  Embedding model: {embedding_model}")
    
    rag_pipeline = RAGPipeline(
        data_dir=data_dir,
        model_path=model_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k
    )
    
    # Index documents
    try:
        rag_pipeline.index_documents()
    except Exception as e:
        print(f"Warning: Could not index documents: {e}")
        print("You may need to index documents manually via /index endpoint")


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    initialize_pipeline()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Sanskrit RAG System API",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "index": "/index",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    if rag_pipeline is None:
        return HealthResponse(
            status="not_initialized",
            indexed=False,
            model_loaded=False
        )
    
    return HealthResponse(
        status="healthy",
        indexed=rag_pipeline.is_indexed,
        model_loaded=rag_pipeline.generator.llm is not None
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    if not rag_pipeline.is_indexed:
        raise HTTPException(status_code=503, detail="Documents not indexed. Please call /index first.")
    
    try:
        # Override top_k if provided
        if request.top_k:
            rag_pipeline.top_k = request.top_k
        
        # Query the pipeline
        result = rag_pipeline.query(request.question, use_rag=request.use_rag)
        
        # Format response
        retrieved_sources = []
        if result.get('retrieved_chunks'):
            for chunk in result['retrieved_chunks']:
                retrieved_sources.append({
                    'source': chunk['metadata']['filename'],
                    'chunk_index': chunk['metadata']['chunk_index'],
                    'score': chunk['score'],
                    'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                })
        
        return QueryResponse(
            answer=result['answer'],
            context=result.get('context'),
            query=result['query'],
            retrieved_sources=retrieved_sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/index")
async def index_documents(force: bool = False):
    """Index documents manually"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        rag_pipeline.index_documents(force_reindex=force)
        return {
            "status": "success",
            "message": "Documents indexed successfully",
            "indexed": rag_pipeline.is_indexed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing documents: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sanskrit RAG System Server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Port to bind to")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "../data"), help="Data directory path")
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH"), help="Path to GGUF model file")
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                       help="Embedding model name")
    
    args = parser.parse_args()
    
    # Set environment variables (override .env with command line args)
    os.environ["DATA_DIR"] = args.data_dir
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    os.environ["EMBEDDING_MODEL"] = args.embedding_model
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
