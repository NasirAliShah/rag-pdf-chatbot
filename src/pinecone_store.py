"""
Pinecone Vector Store Module
=============================
Alternative to FAISS - uses Pinecone cloud vector database.

Advantages of Pinecone:
- Cloud-based (no local storage needed)
- Scales to billions of vectors
- Built-in metadata filtering
- Persistent storage (survives app restarts)
- Fast similarity search

Setup:
1. Sign up at https://www.pinecone.io/
2. Get API key from dashboard
3. Add to .env: PINECONE_API_KEY=your-key
"""

from typing import List, Tuple, Optional
import os
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


class HuggingFaceEmbeddings:
    """Generate embeddings using Hugging Face models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Hugging Face embeddings.
        
        Args:
            model_name: Hugging Face model name
            
        Popular models:
        - sentence-transformers/all-MiniLM-L6-v2 (384 dim, fast)
        - sentence-transformers/all-mpnet-base-v2 (768 dim, better quality)
        - BAAI/bge-small-en-v1.5 (384 dim, good for retrieval)
        - BAAI/bge-base-en-v1.5 (768 dim, better quality)
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Get embedding dimension
        with torch.no_grad():
            test_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
            test_output = self.model(**test_input)
            self.embedding_dim = test_output.last_hidden_state.mean(dim=1).shape[1]
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
        
        Returns:
            Numpy array of embeddings (num_texts, embedding_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding."""
        return self.encode([text])[0]


class PineconeVectorStore:
    """Vector store using Pinecone cloud database."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: str = "rag-pdf-chatbot",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        metric: str = "cosine"
    ):
        """
        Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key (or set PINECONE_API_KEY env var)
            index_name: Name of Pinecone index
            embedding_model: Hugging Face model name
            dimension: Embedding dimension (must match model)
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        # Get API key
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Pinecone API key required. Set PINECONE_API_KEY in .env or pass api_key parameter.\n"
                "Get your API key from: https://www.pinecone.io/"
            )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        # Initialize embeddings generator
        self.embeddings_gen = HuggingFaceEmbeddings(embedding_model)
        
        # Create or connect to index
        self._setup_index()
    
    def _setup_index(self):
        """Create Pinecone index if it doesn't exist."""
        # List existing indexes
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            # Create new index
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
    
    def add_documents(self, chunks: List[str], namespace: str = "default") -> None:
        """
        Add document chunks to Pinecone.
        
        Args:
            chunks: List of text chunks
            namespace: Pinecone namespace (for organizing data)
        """
        # Generate embeddings
        embeddings = self.embeddings_gen.encode(chunks)
        
        # Prepare vectors for upsert
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"chunk_{i}",
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk,
                    "chunk_index": i
                }
            })
        
        # Upsert to Pinecone (batch size 100)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
    
    def search(
        self,
        query: str,
        k: int = 5,
        namespace: str = "default"
    ) -> List[Tuple[str, float]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            namespace: Pinecone namespace
        
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.embeddings_gen.encode_single(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True,
            namespace=namespace
        )
        
        # Extract results
        chunks = []
        for match in results.matches:
            text = match.metadata.get("text", "")
            score = match.score
            chunks.append((text, score))
        
        return chunks
    
    def delete_all(self, namespace: str = "default") -> None:
        """Delete all vectors in namespace."""
        self.index.delete(delete_all=True, namespace=namespace)
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return self.index.describe_index_stats()


class RAGPineconeStore:
    """
    RAG Vector Store using Pinecone.
    
    Drop-in replacement for RAGVectorStore with Pinecone backend.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: str = "rag-pdf-chatbot",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG Pinecone store.
        
        Args:
            api_key: Pinecone API key
            index_name: Pinecone index name
            model_name: Hugging Face embedding model
        """
        # Determine dimension based on model
        dimension_map = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
        }
        dimension = dimension_map.get(model_name, 384)
        
        self.vector_store = PineconeVectorStore(
            api_key=api_key,
            index_name=index_name,
            embedding_model=model_name,
            dimension=dimension
        )
        self.namespace = "default"
    
    def add_documents(self, chunks: List[str]) -> None:
        """Add documents to Pinecone."""
        self.vector_store.add_documents(chunks, namespace=self.namespace)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar chunks."""
        return self.vector_store.search(query, k=k, namespace=self.namespace)
    
    def clear(self) -> None:
        """Clear all data."""
        self.vector_store.delete_all(namespace=self.namespace)
    
    def get_stats(self) -> dict:
        """Get statistics."""
        return self.vector_store.get_stats()
