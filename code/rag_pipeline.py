"""
Complete RAG Pipeline combining Retriever and Generator
"""

from typing import List, Dict, Optional
from document_loader import DocumentLoader, SanskritPreprocessor
from retriever import VectorRetriever
from generator import LLMGenerator


class RAGPipeline:
    """End-to-end RAG pipeline for Sanskrit documents"""
    
    def __init__(
        self,
        data_dir: str = "../data",
        model_path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        top_k: int = 3
    ):
        """
        Initialize RAG Pipeline
        
        Args:
            data_dir: Directory containing Sanskrit documents
            model_path: Path to GGUF model file for llama.cpp
            embedding_model: Name of sentence transformer model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
        """
        self.data_dir = data_dir
        self.top_k = top_k
        
        # Initialize components
        self.loader = DocumentLoader(data_dir)
        self.preprocessor = SanskritPreprocessor(chunk_size, chunk_overlap)
        self.retriever = VectorRetriever(embedding_model)
        # Generator will read n_ctx and n_threads from env if not provided
        self.generator = LLMGenerator(model_path)
        
        # State
        self.is_indexed = False
        self._models_loaded = False
    
    def index_documents(self, force_reindex: bool = False):
        """Index all documents in the data directory"""
        # Try to load existing embeddings
        if not force_reindex and self.retriever.load_embeddings():
            print("Loaded existing embeddings from disk")
            self.is_indexed = True
            return
        
        # Load and preprocess documents
        print("Loading documents...")
        documents = self.loader.load_all_documents()
        
        if not documents:
            raise ValueError(f"No documents found in {self.data_dir}")
        
        print(f"Preprocessing {len(documents)} documents...")
        chunks = self.preprocessor.preprocess_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Create embeddings
        print("Creating embeddings...")
        self.retriever.create_embeddings(chunks)
        
        self.is_indexed = True
        print("Indexing complete!")
        
        # Pre-load models for faster queries
        self._preload_models()
    
    def query(self, question: str, use_rag: bool = True) -> Dict[str, any]:
        """
        Query the RAG system
        
        Args:
            question: User question in Sanskrit or transliterated text
            use_rag: If True, use RAG; if False, use generator only
        
        Returns:
            Dictionary with answer, context, and metadata
        """
        if not self.is_indexed:
            raise ValueError("Documents not indexed. Call index_documents() first.")
        
        if use_rag:
            # Optimized: retrieve and get context in one pass
            context = self.retriever.get_context(question, top_k=self.top_k, max_chars=400)
            
            # Generate response with context
            answer = self.generator.generate_rag_response(question, context)
            
            return {
                'answer': answer,
                'context': None,  # Don't return context for optimization
                'retrieved_chunks': [],  # Don't return chunks for optimization
                'query': question
            }
        else:
            # Generate without context
            answer = self.generator.generate(question, max_tokens=256)
            return {
                'answer': answer,
                'context': None,
                'retrieved_chunks': [],
                'query': question
            }
    
    def _preload_models(self):
        """Pre-load models to avoid loading delay on first query"""
        if not self._models_loaded:
            print("Pre-loading models for faster queries...")
            # Pre-load embedding model
            self.retriever.load_model()
            # Pre-load LLM model if path is set
            if self.generator.model_path:
                try:
                    self.generator.load_model()
                except Exception as e:
                    print(f"Warning: Could not pre-load LLM model: {e}")
            self._models_loaded = True
            print("Models pre-loaded successfully")