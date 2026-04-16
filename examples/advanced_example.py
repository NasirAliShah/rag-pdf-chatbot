"""
Advanced RAG Example
====================
This example demonstrates advanced RAG features:

1. Multi-query retrieval (ask variations of the question)
2. Reranking (reorder results by relevance)
3. Saving/loading vector stores
4. Custom chunking strategies
5. Detailed retrieval analysis

Run this with: python examples/advanced_example.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from src.pdf_loader import PDFLoader
from src.text_chunker import RecursiveCharacterSplitter, TextChunker
from src.embeddings_db import RAGVectorStore
from src.rag_retriever import AdvancedRAGRetriever

load_dotenv()

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * 70)

def example_1_custom_chunking():
    """Example 1: Different chunking strategies."""
    print_section("Example 1: Custom Chunking Strategies")
    
    sample_text = """
    Machine learning is a subset of artificial intelligence.
    It focuses on enabling computers to learn from data.
    Deep learning uses neural networks with multiple layers.
    Natural language processing helps computers understand text.
    Computer vision enables machines to interpret images.
    """
    
    print_subsection("Strategy 1: Character-based chunking")
    chunker1 = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks1 = chunker1.chunk_text(sample_text)
    for i, chunk in enumerate(chunks1, 1):
        print(f"Chunk {i}: {chunk[:50]}...")
    
    print_subsection("Strategy 2: Sentence-based chunking")
    chunker2 = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks2 = chunker2.chunk_text_by_sentences(sample_text)
    for i, chunk in enumerate(chunks2, 1):
        print(f"Chunk {i}: {chunk[:50]}...")
    
    print_subsection("Strategy 3: Recursive character splitting")
    chunker3 = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=20)
    chunks3 = chunker3.split_text(sample_text)
    for i, chunk in enumerate(chunks3, 1):
        print(f"Chunk {i}: {chunk[:50]}...")

def example_2_vector_store_persistence():
    """Example 2: Saving and loading vector stores."""
    print_section("Example 2: Vector Store Persistence")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return
    
    pdf_path = "sample.pdf"
    if not Path(pdf_path).exists():
        print(f"Error: {pdf_path} not found")
        return
    
    print_subsection("Step 1: Load PDF and create vector store")
    loader = PDFLoader(pdf_path)
    text = loader.load()
    
    chunker = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_text(text)
    
    vector_store = RAGVectorStore()
    vector_store.add_documents(chunks)
    print(f"✓ Created vector store with {len(chunks)} chunks")
    
    print_subsection("Step 2: Save vector store to disk")
    save_path = "vector_store_backup"
    vector_store.save(save_path)
    print(f"✓ Saved to {save_path}/")
    
    print_subsection("Step 3: Load vector store from disk")
    loaded_store = RAGVectorStore()
    loaded_store.load(save_path)
    print(f"✓ Loaded vector store")
    
    print_subsection("Step 4: Test loaded store")
    results = loaded_store.search("What is the main topic?", k=3)
    print(f"✓ Retrieved {len(results)} relevant chunks")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n  Result {i} (similarity: {1/(1+score):.1%}):")
        print(f"  {chunk[:100]}...")

def example_3_multi_query_retrieval():
    """Example 3: Multi-query retrieval for better results."""
    print_section("Example 3: Multi-Query Retrieval")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return
    
    pdf_path = "sample.pdf"
    if not Path(pdf_path).exists():
        print(f"Error: {pdf_path} not found")
        return
    
    print_subsection("Setup: Load PDF and create retriever")
    loader = PDFLoader(pdf_path)
    text = loader.load()
    
    chunker = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_text(text)
    
    vector_store = RAGVectorStore()
    vector_store.add_documents(chunks)
    
    retriever = AdvancedRAGRetriever(vector_store, api_key)
    print(f"✓ Retriever ready")
    
    print_subsection("Standard Retrieval")
    question = "What are the main concepts?"
    result1 = retriever.generate_answer(question, k=5, include_sources=False)
    print(f"Answer: {result1['answer'][:200]}...")
    print(f"Sources used: {result1['num_sources']}")
    
    print_subsection("Multi-Query Retrieval")
    result2 = retriever.generate_answer_with_multi_query(
        question,
        k=5,
        include_sources=False
    )
    print(f"Answer: {result2['answer'][:200]}...")
    print(f"Sources used: {result2['num_sources']}")
    
    print("\nMulti-query retrieval:")
    print("- Generates variations of your question")
    print("- Retrieves chunks for each variation")
    print("- Combines results for better coverage")
    print("- Often finds more relevant information")

def example_4_retrieval_analysis():
    """Example 4: Analyze retrieval results in detail."""
    print_section("Example 4: Detailed Retrieval Analysis")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return
    
    pdf_path = "sample.pdf"
    if not Path(pdf_path).exists():
        print(f"Error: {pdf_path} not found")
        return
    
    print_subsection("Setup")
    loader = PDFLoader(pdf_path)
    text = loader.load()
    
    chunker = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_text(text)
    
    vector_store = RAGVectorStore()
    vector_store.add_documents(chunks)
    
    retriever = AdvancedRAGRetriever(vector_store, api_key)
    print(f"✓ Loaded {len(chunks)} chunks")
    
    print_subsection("Analyze Retrieval Results")
    question = "What is the main topic?"
    
    results = retriever.retrieve(question, k=10)
    
    print(f"Question: {question}\n")
    print(f"Retrieved {len(results)} chunks:\n")
    
    for i, (chunk, distance) in enumerate(results, 1):
        similarity = 1 / (1 + distance)
        print(f"Rank {i}: Similarity {similarity:.1%} (distance: {distance:.3f})")
        print(f"  {chunk[:80]}...")
        print()

def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  Advanced RAG Examples")
    print("=" * 70)
    
    example_1_custom_chunking()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        example_2_vector_store_persistence()
        example_3_multi_query_retrieval()
        example_4_retrieval_analysis()
    else:
        print("\n⚠️  Skipping examples 2-4 (require OPENAI_API_KEY)")
    
    print("\n" + "=" * 70)
    print("  Examples Complete!")
    print("=" * 70)
    print("\nFor more information, see SETUP.md")

if __name__ == "__main__":
    main()
