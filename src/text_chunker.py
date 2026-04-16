"""
Text Chunker Module
===================
This module handles the second step of the RAG pipeline: Text → Chunks

What it does:
- Splits large text into smaller, manageable chunks
- Overlaps chunks so context isn't lost between boundaries
- Ensures each chunk is meaningful and self-contained

Why it's needed:
- LLMs have token limits (can't process entire documents at once)
- Embeddings work better with smaller, focused text segments
- Overlapping prevents losing important context at chunk boundaries

How it works:
- Chunk size: How many characters per chunk (e.g., 1000 chars)
- Overlap: How much text repeats between chunks (e.g., 200 chars)
  This ensures context from previous chunk is available in next chunk
"""

from typing import List


class TextChunker:
    """Splits text into overlapping chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Number of characters per chunk (default: 1000)
            chunk_overlap: Number of overlapping characters between chunks (default: 200)
        
        Example:
            If you have text "ABCDEFGHIJ" with chunk_size=4 and overlap=2:
            - Chunk 1: "ABCD"
            - Chunk 2: "CDEF" (overlaps with "CD")
            - Chunk 3: "EFGH" (overlaps with "EF")
            - Chunk 4: "GHIJ" (overlaps with "GH")
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to split
        
        Returns:
            List of text chunks
        """
        if not text or len(text) == 0:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def chunk_text_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks while respecting sentence boundaries.
        
        This is better than character-based chunking because it doesn't
        split sentences in the middle.
        
        Args:
            text: The text to split
        
        Returns:
            List of text chunks
        """
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class RecursiveCharacterSplitter:
    """
    Simple chunker that splits text into fixed-size overlapping chunks.
    
    This is a simplified version that reliably creates multiple chunks
    with proper overlap between them.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to split
        
        Returns:
            List of text chunks
        """
        if not text or len(text) == 0:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk from start to start + chunk_size
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            
            # Move start forward by (chunk_size - overlap)
            # This creates overlap between chunks
            start += self.chunk_size - self.chunk_overlap
            
            # Prevent infinite loop if we're at the end
            if start >= len(text):
                break
        
        return chunks
