"""
RAG PDF Chatbot - Main Application
===================================
This is the main Streamlit application that brings together all RAG components.

How to use:
1. Run: streamlit run app.py
2. Upload a PDF file
3. Ask questions about the PDF content
4. Get answers based on the document

The app handles the complete RAG pipeline:
PDF → Text → Chunks → Embeddings → Vector DB → Retrieval → LLM → Answer
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import tempfile

from src.pdf_loader import PDFLoader
from src.text_chunker import RecursiveCharacterSplitter
from src.embeddings_db import RAGVectorStore
from src.rag_retriever import AdvancedRAGRetriever

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 RAG PDF Chatbot")
st.markdown("""
This application demonstrates a **Retrieval-Augmented Generation (RAG)** system.
Upload a PDF and ask questions about its content!
""")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=200,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of text chunks. Larger = more context, smaller = more specific"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Overlap between chunks to preserve context"
    )
    
    num_results = st.slider(
        "Number of Context Chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant chunks to use for answering"
    )
    
    st.divider()
    st.header("📚 About RAG")
    st.markdown("""
    **RAG (Retrieval-Augmented Generation)** combines:
    
    1. **Retrieval**: Find relevant information from documents
    2. **Augmentation**: Add this info to the LLM prompt
    3. **Generation**: LLM generates answer based on context
    
    This prevents hallucination and grounds answers in facts!
    """)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OPENAI_API_KEY not found in .env file")
    st.info("Please create a .env file with: OPENAI_API_KEY=your_key_here")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to ask questions about"
    )

with col2:
    st.header("🔄 Status")
    status_placeholder = st.empty()

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / uploaded_file.name
        pdf_path.write_bytes(uploaded_file.getvalue())
        
        with status_placeholder.container():
            st.info("📖 Loading PDF...")
        
        try:
            pdf_loader = PDFLoader(str(pdf_path))
            pdf_data = pdf_loader.load_with_metadata()
            
            with status_placeholder.container():
                st.success(f"✅ Loaded {pdf_data['num_pages']} pages")
            
            st.subheader(f"📋 Document: {pdf_data['filename']}")
            st.caption(f"Pages: {pdf_data['num_pages']} | Characters: {len(pdf_data['text']):,}")
            
            with st.expander("👁️ Preview first 500 characters"):
                st.text(pdf_data['text'][:500] + "...")
            
            st.divider()
            
            with status_placeholder.container():
                st.info("✂️ Chunking text...")
            
            chunker = RecursiveCharacterSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = chunker.split_text(pdf_data['text'])
            
            with status_placeholder.container():
                st.success(f"✅ Created {len(chunks)} chunks")
            
            st.subheader("📊 Chunking Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", len(chunks))
            with col2:
                avg_chunk_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
                st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f} chars")
            with col3:
                st.metric("Total Characters", sum(len(c) for c in chunks))
            
            with st.expander("👁️ View sample chunks"):
                for i, chunk in enumerate(chunks[:3]):
                    st.write(f"**Chunk {i+1}:**")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                    st.divider()
            
            st.divider()
            
            with status_placeholder.container():
                st.info("🔢 Generating embeddings...")
            
            vector_store = RAGVectorStore(model_name="all-MiniLM-L6-v2")
            vector_store.add_documents(chunks)
            
            with status_placeholder.container():
                st.success("✅ Embeddings ready")
            
            retriever = AdvancedRAGRetriever(vector_store, api_key)
            
            st.subheader("❓ Ask Questions")
            st.markdown("Ask anything about the document content:")
            
            question = st.text_input(
                "Your question:",
                placeholder="e.g., What is the main topic of this document?",
                label_visibility="collapsed"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                search_button = st.button("🔍 Search & Answer", use_container_width=True)
            with col2:
                retrieval_method = st.selectbox(
                    "Method",
                    ["Standard", "Multi-Query"],
                    label_visibility="collapsed"
                )
            with col3:
                show_sources = st.checkbox("Show Sources", value=True)
            
            if search_button and question:
                with st.spinner("🤔 Thinking..."):
                    if retrieval_method == "Multi-Query":
                        result = retriever.generate_answer_with_multi_query(
                            question,
                            k=num_results,
                            include_sources=show_sources
                        )
                    else:
                        result = retriever.generate_answer(
                            question,
                            k=num_results,
                            include_sources=show_sources
                        )
                
                st.subheader("💡 Answer")
                st.write(result['answer'])
                
                st.subheader(f"📚 Sources Used ({result['num_sources']})")
                
                if show_sources and result.get('sources'):
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(f"Source {i}"):
                            st.text(source)
                else:
                    st.caption("Sources not included in response")
                
                st.divider()
                st.success("✅ Answer generated successfully!")
            
            elif search_button:
                st.warning("Please enter a question first!")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

else:
    st.info("👆 Upload a PDF file to get started!")
    
    with st.expander("📖 How RAG Works"):
        st.markdown("""
        ### The RAG Pipeline
        
        **Step 1: PDF → Text**
        - Extract text from PDF file
        - Preserve document structure
        
        **Step 2: Text → Chunks**
        - Split text into manageable pieces
        - Use overlapping to preserve context
        - Example: 1000 char chunks with 200 char overlap
        
        **Step 3: Chunks → Embeddings**
        - Convert each chunk to a numerical vector
        - Similar text = similar vectors
        - Uses sentence transformer model
        
        **Step 4: Embeddings → Vector DB**
        - Store all embeddings in FAISS database
        - Enables fast similarity search
        - Can search thousands of chunks instantly
        
        **Step 5: User Question → Embedding**
        - Convert user's question to embedding
        - Same model as chunks for consistency
        
        **Step 6: Similar Chunks → Retrieval**
        - Find most similar chunks to question
        - Use vector similarity (L2 distance)
        - Return top-k most relevant chunks
        
        **Step 7: Chunks → LLM → Answer**
        - Send relevant chunks + question to LLM
        - LLM generates answer based on context
        - Prevents hallucination, grounds in facts
        """)
    
    with st.expander("⚙️ Configuration Tips"):
        st.markdown("""
        **Chunk Size:**
        - Smaller (200-500): More specific, less context
        - Medium (800-1200): Balanced (recommended)
        - Larger (1500+): More context, less specific
        
        **Chunk Overlap:**
        - Prevents losing context at chunk boundaries
        - 15-20% of chunk size is typical
        - Example: 1000 char chunks → 150-200 overlap
        
        **Number of Results:**
        - More results = more context but slower
        - 3-5: Fast, focused answers
        - 5-10: Comprehensive answers
        """)
