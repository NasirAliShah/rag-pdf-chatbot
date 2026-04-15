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
    More advanced chunker that splits by different separators.
    
    Instead of just splitting by character count, it tries to split by:
    1. Paragraphs (double newlines)
    2. Sentences (periods)
    3. Words (spaces)
    4. Characters (last resort)
    
    This keeps related content together better.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text using recursive character splitting.
        
        Args:
            text: The text to split
        
        Returns:
            List of text chunks
        """
        final_chunks = []
        separator = self.separators[-1]
        
        for _s in self.separators:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                other_info = self.split_text(s)
                final_chunks.extend(other_info)
        
        if good_splits:
            merged_text = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged_text)
        
        return final_chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks of appropriate size."""
        separator_len = len(separator)
        good_splits = []
        
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = separator.join(good_splits)
                    if len(merged_text) > self.chunk_size:
                        if good_splits:
                            merged_text = self._merge_splits(good_splits, separator)
                            good_splits = []
                good_splits.append(s)
        
        if good_splits:
            merged_text = separator.join(good_splits)
            return [merged_text]
        
        return []
