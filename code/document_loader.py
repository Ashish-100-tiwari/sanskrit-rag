"""
Document Loader and Preprocessor for Sanskrit Documents
Supports .txt, .pdf, and .docx formats
"""

import os
from typing import List, Dict
from pathlib import Path
import re


class DocumentLoader:
    """Load and preprocess Sanskrit documents from various formats"""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.documents = []
        
    def load_txt(self, file_path: Path) -> str:
        """Load text from .txt file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except:
                    continue
            raise ValueError(f"Could not decode {file_path}")
    
    def load_pdf(self, file_path: Path) -> str:
        """Load text from .pdf file using PyPDF2"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
    
    def load_docx(self, file_path: Path) -> str:
        """Load text from .docx file"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except ImportError:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")
    
    def load_document(self, file_path: Path) -> str:
        """Load document based on file extension"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return self.load_txt(file_path)
        elif suffix == '.pdf':
            return self.load_pdf(file_path)
        elif suffix == '.docx':
            return self.load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def load_all_documents(self) -> List[Dict[str, str]]:
        """Load all documents from data directory"""
        documents = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Supported file extensions
        extensions = ['.txt', '.pdf', '.docx']
        
        for file_path in self.data_dir.iterdir():
            if file_path.suffix.lower() in extensions:
                try:
                    content = self.load_document(file_path)
                    documents.append({
                        'filename': file_path.name,
                        'content': content,
                        'filepath': str(file_path)
                    })
                    print(f"Loaded: {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
        
        self.documents = documents
        return documents


class SanskritPreprocessor:
    """Preprocess Sanskrit text for RAG pipeline"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Sanskrit text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but preserve Sanskrit Unicode
        # Keep Devanagari script (U+0900 to U+097F) and common punctuation
        text = text.strip()
        return text
    
    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean the text first
        text = self.clean_text(text)
        
        # Split by sentences (period, exclamation, question mark)
        sentences = re.split(r'[।॥.!?]\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Take last few sentences for overlap
                    overlap_sentences = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        # If no chunks created (text is shorter than chunk_size), return full text
        if not chunks:
            chunks = [text] if text else []
        
        return chunks
    
    def preprocess_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """Preprocess all documents into chunks"""
        processed_chunks = []
        
        for doc in documents:
            chunks = self.split_into_chunks(doc['content'])
            
            for idx, chunk in enumerate(chunks):
                processed_chunks.append({
                    'text': chunk,
                    'source': doc['filename'],
                    'chunk_id': idx,
                    'metadata': {
                        'filename': doc['filename'],
                        'chunk_index': idx,
                        'total_chunks': len(chunks)
                    }
                })
        
        return processed_chunks
