"""
Vector-based Retriever for Sanskrit Documents
Uses sentence transformers for embeddings
"""

import os
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path


class VectorRetriever:
    """Vector-based retriever using embeddings"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize retriever with embedding model
        Using multilingual model that supports Sanskrit Unicode
        """
        self.model_name = model_name
        self.embedding_model = None
        self.embeddings = None
        self.chunks = []
        self.vector_store_path = Path("../data/vector_store.npz")
        
    def load_model(self):
        """Load the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            print("Model loaded successfully")
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
    def create_embeddings(self, chunks: List[Dict[str, any]]):
        """Create embeddings for all chunks"""
        if self.embedding_model is None:
            self.load_model()
        
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"Creating embeddings for {len(texts)} chunks...")
        self.embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"Created embeddings with shape: {self.embeddings.shape}")
        
        # Save embeddings
        self.save_embeddings()
    
    def save_embeddings(self):
        """Save embeddings to disk"""
        if self.embeddings is not None and self.chunks:
            # Save embeddings
            np.savez_compressed(
                self.vector_store_path,
                embeddings=self.embeddings,
                chunks=[chunk['text'] for chunk in self.chunks],
                metadata=[chunk['metadata'] for chunk in self.chunks]
            )
            print(f"Saved embeddings to {self.vector_store_path}")
    
    def load_embeddings(self):
        """Load embeddings from disk"""
        if self.vector_store_path.exists():
            print(f"Loading embeddings from {self.vector_store_path}")
            data = np.load(self.vector_store_path, allow_pickle=True)
            self.embeddings = data['embeddings']
            
            # Reconstruct chunks
            chunks_text = data['chunks']
            metadata = data['metadata']
            self.chunks = [
                {
                    'text': text,
                    'metadata': meta
                }
                for text, meta in zip(chunks_text, metadata)
            ]
            print(f"Loaded {len(self.chunks)} chunks")
            return True
        return False
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """Retrieve top-k most relevant chunks for a query"""
        if self.embedding_model is None:
            self.load_model()
        
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("No embeddings found. Please create embeddings first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Retrieve top-k chunks with scores
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx]['text'],
                'score': float(similarities[idx]),
                'metadata': self.chunks[idx]['metadata']
            })
        
        return results
    
    def get_context(self, query: str, top_k: int = 5) -> str:
        """Get formatted context string from retrieved chunks"""
        results = self.retrieve(query, top_k)
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source: {result['metadata']['filename']}, Chunk {result['metadata']['chunk_index']+1}]\n"
                f"{result['text']}\n"
            )
        
        return "\n".join(context_parts)
