"""
RAG Retriever and LLM Integration Module
=========================================
This module handles the final steps of the RAG pipeline:
User Question → Embedding → Similar Chunks → LLM → Answer

What it does:
- Takes user questions and converts them to embeddings
- Retrieves the most relevant chunks from vector database
- Sends relevant chunks + question to LLM for answer generation
- Formats the context for the LLM

Why it's needed:
- LLMs are powerful but can hallucinate (make up information)
- By providing relevant context from the PDF, we ground the answer in facts
- This makes the LLM more accurate and trustworthy

How it works:
1. User asks: "What is machine learning?"
2. Convert question to embedding
3. Search vector DB for similar chunks (e.g., chunks about ML from PDF)
4. Create a prompt with: context chunks + user question
5. Send to LLM which generates answer based on provided context
"""

from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .embeddings_db import RAGVectorStore, EmbeddingsGenerator


class RAGRetriever:
    """Retrieves relevant chunks and generates answers using LLM."""
    
    def __init__(self, vector_store: RAGVectorStore, api_key: str):
        """
        Initialize the RAG retriever.
        
        Args:
            vector_store: RAGVectorStore instance with loaded documents
            api_key: OpenAI API key
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.7
        )
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User's question or search query
            k: Number of chunks to retrieve
        
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        return self.vector_store.search(query, k)
    
    def generate_answer(
        self,
        query: str,
        k: int = 5,
        include_sources: bool = True
    ) -> dict:
        """
        Generate an answer to a query using retrieved context.
        
        Args:
            query: User's question
            k: Number of context chunks to use
            include_sources: Whether to include source chunks in response
        
        Returns:
            Dictionary containing:
            - 'answer': The generated answer
            - 'sources': List of source chunks used (if include_sources=True)
            - 'num_sources': Number of sources used
        """
        retrieved_chunks = self.retrieve(query, k)
        
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find relevant information in the document.",
                'sources': [],
                'num_sources': 0
            }
        
        context = self._format_context(retrieved_chunks)
        
        system_prompt = """You are a helpful assistant that answers questions based on provided context.
Always answer based on the context given. If the context doesn't contain the answer, say so clearly.
Be concise and accurate."""
        
        user_prompt = f"""Context from document:
{context}

Question: {query}

Please answer the question based on the context provided above."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        result = {
            'answer': answer,
            'num_sources': len(retrieved_chunks)
        }
        
        if include_sources:
            result['sources'] = [chunk for chunk, _ in retrieved_chunks]
        
        return result
    
    def _format_context(self, chunks: List[Tuple[str, float]]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        
        Args:
            chunks: List of (chunk_text, similarity_score) tuples
        
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, (chunk, score) in enumerate(chunks, 1):
            context_parts.append(f"[Source {i}] (Relevance: {1/(1+score):.2%})\n{chunk}\n")
        
        return "\n".join(context_parts)


class AdvancedRAGRetriever(RAGRetriever):
    """
    Advanced RAG retriever with additional features:
    - Multi-query retrieval (ask multiple variations of the question)
    - Reranking (reorder results by relevance)
    - Context compression (remove irrelevant parts from context)
    """
    
    def generate_answer_with_reranking(
        self,
        query: str,
        k: int = 10,
        rerank_k: int = 5,
        include_sources: bool = True
    ) -> dict:
        """
        Generate answer with reranking for better relevance.
        
        This retrieves more chunks initially (k=10) then reranks them
        to select the most relevant ones (rerank_k=5).
        
        Args:
            query: User's question
            k: Initial number of chunks to retrieve
            rerank_k: Number of chunks to use after reranking
            include_sources: Whether to include source chunks
        
        Returns:
            Dictionary with answer and sources
        """
        retrieved_chunks = self.retrieve(query, k)
        
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find relevant information in the document.",
                'sources': [],
                'num_sources': 0
            }
        
        reranked_chunks = retrieved_chunks[:rerank_k]
        
        context = self._format_context(reranked_chunks)
        
        system_prompt = """You are a helpful assistant that answers questions based on provided context.
Always answer based on the context given. If the context doesn't contain the answer, say so clearly.
Be concise and accurate."""
        
        user_prompt = f"""Context from document:
{context}

Question: {query}

Please answer the question based on the context provided above."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        result = {
            'answer': answer,
            'num_sources': len(reranked_chunks)
        }
        
        if include_sources:
            result['sources'] = [chunk for chunk, _ in reranked_chunks]
        
        return result
    
    def generate_answer_with_multi_query(
        self,
        query: str,
        k: int = 5,
        include_sources: bool = True
    ) -> dict:
        """
        Generate answer using multiple query variations.
        
        This asks the LLM to generate variations of the question,
        retrieves chunks for each variation, and combines results.
        
        Args:
            query: User's question
            k: Number of chunks per query variation
            include_sources: Whether to include source chunks
        
        Returns:
            Dictionary with answer and sources
        """
        query_variations = self._generate_query_variations(query)
        
        all_chunks = {}
        for q in query_variations:
            chunks = self.retrieve(q, k)
            for chunk, score in chunks:
                if chunk not in all_chunks:
                    all_chunks[chunk] = score
                else:
                    all_chunks[chunk] = min(all_chunks[chunk], score)
        
        sorted_chunks = sorted(all_chunks.items(), key=lambda x: x[1])[:k]
        
        if not sorted_chunks:
            return {
                'answer': "I couldn't find relevant information in the document.",
                'sources': [],
                'num_sources': 0
            }
        
        context = self._format_context(sorted_chunks)
        
        system_prompt = """You are a helpful assistant that answers questions based on provided context.
Always answer based on the context given. If the context doesn't contain the answer, say so clearly.
Be concise and accurate."""
        
        user_prompt = f"""Context from document:
{context}

Question: {query}

Please answer the question based on the context provided above."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        result = {
            'answer': answer,
            'num_sources': len(sorted_chunks)
        }
        
        if include_sources:
            result['sources'] = [chunk for chunk, _ in sorted_chunks]
        
        return result
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """
        Generate variations of the user's query.
        
        This helps find relevant chunks that might use different wording.
        """
        system_prompt = """Generate 3 different variations of the given question that could help find relevant information. 
Return only the questions, one per line, without numbering."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original question: {query}")
        ]
        
        response = self.llm.invoke(messages)
        variations = response.content.strip().split('\n')
        variations = [v.strip() for v in variations if v.strip()]
        
        return [query] + variations
