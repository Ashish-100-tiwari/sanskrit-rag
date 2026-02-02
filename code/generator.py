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
            n_ctx: Context window size (defaults to env var or 512 for optimization)
            n_threads: Number of CPU threads to use (defaults to env var or 4)
        """
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.n_ctx = n_ctx or int(os.getenv("N_CTX", "512"))
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
            # Highly optimized settings for speed
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=512,  # Larger batch for faster processing
                n_gpu_layers=0,  # CPU only
                verbose=False,
                use_mmap=True,  # Memory mapping for faster loading
                use_mlock=False  # Don't lock memory
            )
            print("Model loaded successfully")
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required. Install with: pip install llama-cpp-python\n"
                "For CPU-only builds: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            )
    
    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.5, top_p: float = 0.9) -> str:
        """Generate text from prompt - highly optimized for speed"""
        if self.llm is None:
            self.load_model()
        
        # Highly optimized generation parameters for minimal latency
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=20,  # Limit top_k for faster sampling
            echo=False,
            stop=["\n\n", "Human:", "Question:", "\n"],
            repeat_penalty=1.1,
            tfs_z=1.0,  # Tail free sampling
            mirostat_mode=0  # Disable mirostat for speed
        )
        
        return response['choices'][0]['text'].strip()
    
    def generate_rag_response(self, query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """Generate RAG response with context - highly optimized for speed"""
        # Limit context to 200 chars for minimal latency
        max_context_chars = 200
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "..."
        
        # Ultra-compact prompt for fastest generation
        prompt = f"{context}\nQ: {query}\nA:"
        
        # Reduced tokens and temperature for speed
        return self.generate(prompt, max_tokens=128, temperature=0.4)
