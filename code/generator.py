"""
LLM Generator using llama.cpp for CPU-based inference
"""

import os
from typing import Optional
from pathlib import Path


class LLMGenerator:
    """Generator using llama.cpp for CPU inference"""
    
    def __init__(self, model_path: Optional[str] = None, n_ctx: Optional[int] = None, n_threads: Optional[int] = None):
        """
        Initialize LLM generator
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (defaults to env var or 2048)
            n_threads: Number of CPU threads to use (defaults to env var or 4)
        """
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.n_ctx = n_ctx or int(os.getenv("N_CTX", "2048"))
        self.n_threads = n_threads or int(os.getenv("N_THREADS", "4"))
        self.llm = None
        
    def load_model(self):
        """Load the LLM model using llama-cpp-python"""
        if self.model_path is None:
            raise ValueError("Model path not specified. Please set model_path or download a model.")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            from llama_cpp import Llama
            print(f"Loading model from {self.model_path}...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            print("Model loaded successfully")
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required. Install with: pip install llama-cpp-python\n"
                "For CPU-only builds: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            )
    
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate text from prompt"""
        if self.llm is None:
            self.load_model()
        
        # Create the full prompt
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
            stop=["\n\n", "Human:", "Question:"]
        )
        
        return response['choices'][0]['text'].strip()
    
    def generate_rag_response(self, query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """Generate RAG response with context"""
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided Sanskrit document context. 
Answer in a clear and informative manner. If the context doesn't contain enough information, say so."""
        
        # Construct the prompt
        prompt = f"""{system_prompt}

Context from Sanskrit documents:
{context}

Question: {query}

Answer:"""
        
        return self.generate(prompt, max_tokens=512, temperature=0.7)
