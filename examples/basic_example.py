"""
Basic RAG Example
=================
This example demonstrates the simplest way to use the RAG system.

It shows:
1. Loading a PDF
2. Chunking the text
3. Creating embeddings
4. Asking questions and getting answers

Run this with: python examples/basic_example.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from src.pdf_loader import PDFLoader
from src.text_chunker import RecursiveCharacterSplitter
from src.embeddings_db import RAGVectorStore
from src.rag_retriever import RAGRetriever

load_dotenv()

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        return
    
    pdf_path = "sample.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Error: {pdf_path} not found")
        print("Please provide a PDF file named 'sample.pdf' in the project root")
        return
    
    print("=" * 60)
    print("RAG PDF Chatbot - Basic Example")
    print("=" * 60)
    
    print("\n[Step 1] Loading PDF...")
    loader = PDFLoader(pdf_path)
    pdf_data = loader.load_with_metadata()
    print(f"✓ Loaded {pdf_data['num_pages']} pages")
    print(f"  Total characters: {len(pdf_data['text']):,}")
    
    print("\n[Step 2] Chunking text...")
    chunker = RecursiveCharacterSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = chunker.split_text(pdf_data['text'])
    print(f"✓ Created {len(chunks)} chunks")
    print(f"  Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")
    
    print("\n[Step 3] Generating embeddings...")
    vector_store = RAGVectorStore(model_name="all-MiniLM-L6-v2")
    vector_store.add_documents(chunks)
    print(f"✓ Embeddings generated and indexed")
    
    print("\n[Step 4] Creating retriever...")
    retriever = RAGRetriever(vector_store, api_key)
    print(f"✓ Retriever ready")
    
    print("\n" + "=" * 60)
    print("Ask Questions About Your PDF")
    print("=" * 60)
    print("(Type 'quit' to exit)\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not question:
            print("Please enter a question.\n")
            continue
        
        print("\n🔍 Searching and generating answer...")
        result = retriever.generate_answer(question, k=5, include_sources=True)
        
        print("\n" + "-" * 60)
        print("ANSWER:")
        print("-" * 60)
        print(result['answer'])
        
        print("\n" + "-" * 60)
        print(f"SOURCES ({result['num_sources']} chunks used):")
        print("-" * 60)
        for i, source in enumerate(result['sources'], 1):
            print(f"\n[Source {i}]")
            print(source[:300] + "..." if len(source) > 300 else source)
        
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
