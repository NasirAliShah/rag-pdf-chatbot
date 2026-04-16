"""
Embeddings and Vector Database Module
======================================
This module handles the third step of the RAG pipeline: Chunks → Embeddings → Vector DB

What it does:
- Converts text chunks into numerical vectors (embeddings)
- Stores these vectors in a vector database (FAISS)
- Allows fast similarity search to find relevant chunks

Why it's needed:
- Embeddings are numerical representations of text meaning
- Similar texts have similar embeddings (close in vector space)
- Vector databases enable fast similarity search (much faster than comparing all chunks)

How embeddings work:
- A sentence like "The cat sat on the mat" becomes a vector like [0.2, -0.5, 0.8, ...]
- Another sentence "A cat was sitting on a mat" has a similar vector
- We can calculate distance between vectors to find similarity
- Smaller distance = more similar meaning

Vector Database (FAISS):
- Stores all chunk embeddings
- Enables fast nearest neighbor search
- When user asks a question, we find the most similar chunks
"""

from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path


class EmbeddingsGenerator:
    """Generates embeddings for text chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embeddings generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
            
        Available models (trade-off between speed and quality):
        - "all-MiniLM-L6-v2": Fast, good quality (recommended for most cases)
        - "all-mpnet-base-v2": Slower, better quality
        - "all-distilroberta-v1": Fast, decent quality
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_embedding_dimension()
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Convert text chunks to embeddings.
        
        Args:
            texts: List of text chunks to encode
        
        Returns:
            2D numpy array where each row is an embedding vector
            Shape: (num_texts, embedding_dimension)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Convert a single text to embedding.
        
        Args:
            text: Text to encode
        
        Returns:
            1D numpy array representing the embedding
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)


class VectorDatabase:
    """Stores and retrieves embeddings using FAISS."""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the vector database.
        
        Args:
            embedding_dim: Dimension of embeddings (must match embedding model)
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = []
        self.embeddings = None
    
    def add_chunks(self, chunks: List[str], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the database.
        
        Args:
            chunks: List of text chunks
            embeddings: 2D numpy array of embeddings (one per chunk)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        self.chunks.extend(chunks)
        self.index.add(embeddings)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the k most similar chunks to a query.
        
        Args:
            query_embedding: 1D numpy array representing the query embedding
            k: Number of results to return
        
        Returns:
            List of tuples (chunk_text, similarity_score)
            Lower score = more similar (FAISS uses L2 distance)
        """
        if self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(distance)))
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save the vector database to disk.
        
        Args:
            path: Directory path to save the database
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        with open(path / "embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)
    
    def load(self, path: str) -> None:
        """
        Load the vector database from disk.
        
        Args:
            path: Directory path where the database is saved
        """
        path = Path(path)
        
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        
        self.embeddings = np.load(path / "embeddings.npy")


class RAGVectorStore:
    """
    Complete vector store combining embeddings generator and database.
    
    This is the main class you'll use for the embedding step of RAG.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG vector store.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.embeddings_gen = EmbeddingsGenerator(model_name)
        self.vector_db = VectorDatabase(self.embeddings_gen.embedding_dim)
    
    def add_documents(self, chunks: List[str]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of text chunks to add
        """
        embeddings = self.embeddings_gen.encode(chunks)
        self.vector_db.add_chunks(chunks, embeddings)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for the most relevant chunks to a query.
        
        Args:
            query: The search query
            k: Number of results to return
        
        Returns:
            List of tuples (chunk_text, similarity_score)
        """
        query_embedding = self.embeddings_gen.encode_single(query)
        return self.vector_db.search(query_embedding, k)
    
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        self.vector_db.save(path)
    
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        self.vector_db.load(path)
