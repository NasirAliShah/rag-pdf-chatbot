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
from src.pinecone_store import RAGPineconeStore
from src.rag_retriever import AdvancedRAGRetriever
from src.token_counter import RAGTokenTracker
from src.chat_history import ChatHistory, ContextualRAGRetriever
from src.pdf_metadata import PDFMetadataManager

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 RAG PDF Chatbot")
st.markdown("""
This application demonstrates a **Retrieval-Augmented Generation (RAG)** system with conversation history.
Upload a PDF and ask questions about its content. The chatbot learns from conversation history!
""")

# Initialize session state for chat history and metadata
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatHistory(max_history=20)
if "conversation_messages" not in st.session_state:
    st.session_state.conversation_messages = []
if "pdf_metadata_manager" not in st.session_state:
    st.session_state.pdf_metadata_manager = PDFMetadataManager()
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("� PDF Management")
    
    # Show previously uploaded PDFs (only for Pinecone)
    pdf_list = st.session_state.pdf_metadata_manager.list_pdfs()
    if pdf_list:
        st.write("**Previously Uploaded PDFs:**")
        for pdf in pdf_list:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📄 {pdf.filename} ({pdf.num_chunks} chunks)")
            with col2:
                if st.button("Load", key=f"load_{pdf.filename}"):
                    st.session_state.current_pdf = pdf.filename
                    st.session_state.conversation_messages = []
                    st.session_state.chat_history.clear()
                    st.rerun()
        st.divider()
    
    st.write("**Upload New PDF:**")
    
    st.subheader("�️ Vector Database")
    vector_db_type = st.radio(
        "Choose Vector Database",
        ["FAISS (Local, Free)", "Pinecone (Cloud, Paid)"],
        help="FAISS stores vectors locally. Pinecone is cloud-based and persistent."
    )
    
    if "Pinecone" in vector_db_type:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            st.error("⚠️ PINECONE_API_KEY not found in .env file")
            st.info("Get your API key from: https://www.pinecone.io/\nAdd to .env: PINECONE_API_KEY=your_key_here")
            st.stop()
        
        embedding_model = st.selectbox(
            "Hugging Face Embedding Model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5",
                "BAAI/bge-base-en-v1.5"
            ],
            help="Choose embedding model from Hugging Face"
        )
    
    st.divider()
    
    st.subheader("🤖 LLM Provider")
    llm_provider = st.radio(
        "Choose LLM Provider",
        ["Ollama (Free, Local)", "OpenAI (Paid, Cloud)"],
        help="Ollama runs locally and is free. OpenAI requires an API key and credits."
    )
    
    if "Ollama" in llm_provider:
        ollama_model = st.selectbox(
            "Ollama Model",
            ["mistral", "llama2", "neural-chat", "dolphin-mixtral"],
            help="Choose a model. mistral is recommended for speed."
        )
        ollama_url = st.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            help="URL where Ollama is running"
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ OPENAI_API_KEY not found in .env file")
            st.info("Please create a .env file with: OPENAI_API_KEY=your_key_here")
            st.stop()
        
        openai_model = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            help="Choose OpenAI model. GPT-3.5 is fastest and cheapest."
        )
    
    st.divider()
    
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

# Check if a PDF is loaded (either uploaded or from Load button)
if uploaded_file is not None or st.session_state.current_pdf:
    is_new_upload = uploaded_file is not None
    pdf_filename = uploaded_file.name if uploaded_file else st.session_state.current_pdf
    chunks = None
    
    # For new uploads, extract and chunk the PDF
    if is_new_upload:
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
                
                # Initialize vector store based on selection
                if "Pinecone" in vector_db_type:
                    vector_store = RAGPineconeStore(
                        api_key=pinecone_api_key,
                        index_name="rag-pdf-chatbot",
                        model_name=embedding_model
                    )
                else:
                    vector_store = RAGVectorStore(model_name="all-MiniLM-L6-v2")
                
                vector_store.add_documents(chunks)
                
                # Save PDF metadata if using Pinecone
                if "Pinecone" in vector_db_type:
                    st.session_state.pdf_metadata_manager.add_pdf(
                        filename=uploaded_file.name,
                        namespace="default",
                        num_chunks=len(chunks)
                    )
                
                with status_placeholder.container():
                    if "Pinecone" in vector_db_type:
                        st.success("✅ Embeddings stored in Pinecone")
                    else:
                        st.success("✅ Embeddings ready")
            
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.stop()
    else:
        # Loading from Pinecone - just initialize vector store
        if "Pinecone" in vector_db_type:
            with status_placeholder.container():
                st.info("📚 Loading from Pinecone...")
            
            try:
                vector_store = RAGPineconeStore(
                    api_key=pinecone_api_key,
                    index_name="rag-pdf-chatbot",
                    model_name=embedding_model
                )
                with status_placeholder.container():
                    st.success(f"✅ Loaded: {pdf_filename}")
            except Exception as e:
                st.error(f"❌ Error connecting to Pinecone: {str(e)}")
                st.stop()
        else:
            st.error(
                "❌ **Cannot Load PDF**\n\n"
                f"The PDF '{pdf_filename}' was uploaded to Pinecone, but you have FAISS selected.\n\n"
                "**To load this PDF:**\n"
                "1. Select 'Pinecone (Cloud, Paid)' in Vector Database\n"
                "2. Click the Load button again"
            )
            st.stop()
    
    # Initialize retriever with selected provider (for both new and loaded PDFs)
    if "Ollama" in llm_provider:
        retriever = AdvancedRAGRetriever(
            vector_store,
            provider="ollama",
            model=ollama_model,
            base_url=ollama_url
        )
    else:
        retriever = AdvancedRAGRetriever(
            vector_store,
            provider="openai",
            api_key=api_key,
            model=openai_model
        )
    
    st.subheader("💬 Chat with Your Document")
    st.markdown("Ask questions and the chatbot will learn from the conversation history!")
    
    # Settings row
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["Standard", "Multi-Query"],
            label_visibility="collapsed"
        )
    with col2:
        show_sources = st.checkbox("Show Sources", value=True)
    with col3:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history.clear()
            st.session_state.conversation_messages = []
            st.rerun()
    
    # Display chat history
    for msg in st.session_state.conversation_messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
    
    # Chat input (single click, no Enter needed)
    question = st.chat_input(
        "Ask a question about the document...",
        key="chat_input"
    )
    
    if question:
        with st.spinner("🤔 Thinking..."):
            try:
                # Initialize token tracker for OpenAI
                token_tracker = None
                if "OpenAI" in llm_provider and chunks:
                    token_tracker = RAGTokenTracker(model="gpt-3.5-turbo")
                    embeddings_tokens = token_tracker.track_embeddings(chunks)
                
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
                
                # Add to chat history
                st.session_state.chat_history.add_message("user", question)
                st.session_state.chat_history.add_message(
                    "assistant",
                    result['answer'],
                    sources=result.get('sources', [])
                )
                
                # Update conversation messages for display
                st.session_state.conversation_messages.append({
                    "role": "user",
                    "content": question
                })
                st.session_state.conversation_messages.append({
                    "role": "assistant",
                    "content": result['answer']
                })
                
                # Display answer in chat format
                st.chat_message("user").write(question)
                st.chat_message("assistant").write(result['answer'])
                
                st.subheader(f"📚 Sources Used ({result['num_sources']})")
                
                if show_sources and result.get('sources'):
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(f"Source {i}"):
                            st.text(source)
                else:
                    st.caption("Sources not included in response")
                
                st.divider()
                
                # Display token usage for OpenAI
                if token_tracker and "OpenAI" in llm_provider:
                    # Track retrieval tokens
                    retrieval_tokens = token_tracker.track_retrieval(
                        question,
                        result.get('sources', [])
                    )
                    
                    # Estimate LLM tokens (rough estimate)
                    context_text = "\n".join(result.get('sources', []))
                    llm_input = f"System: Extract info\n\nContext: {context_text}\n\nQuestion: {question}"
                    llm_output = result['answer']
                    
                    input_tokens, output_tokens, cost = token_tracker.track_llm_call(
                        llm_input,
                        llm_output
                    )
                    
                    # Display token breakdown
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Input Tokens", f"{input_tokens:,}")
                    with col2:
                        st.metric("📤 Output Tokens", f"{output_tokens:,}")
                    with col3:
                        st.metric("💰 Cost", f"${cost:.6f}")
                    
                    # Show detailed breakdown
                    with st.expander("📈 Detailed Token Usage"):
                        summary = token_tracker.get_summary()
                        st.write(f"""
**Token Breakdown:**
- Embeddings (PDF chunks): {summary['embeddings_tokens']:,} tokens
- Retrieval (context): {summary['retrieval_tokens']:,} tokens
- LLM Input: {summary['llm_input_tokens']:,} tokens
- LLM Output: {summary['llm_output_tokens']:,} tokens
- **Total: {summary['total_tokens']:,} tokens**

**Cost Breakdown:**
- Total Cost: **${summary['total_cost']:.6f}**
- Cost per question: **${cost:.6f}**
- Questions per $1: **{int(1 / max(cost, 0.0001))}**
- Questions per $10: **{int(10 / max(cost, 0.0001))}**
                        """)
                
                st.success("✅ Answer generated successfully!")
            
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                    st.error(
                        "❌ **API Quota Exceeded**\n\n"
                        "Your OpenAI API account has reached its usage limit or billing issue.\n\n"
                        "**Solutions:**\n"
                        "1. Check your OpenAI account: https://platform.openai.com/account/billing/overview\n"
                        "2. Add a payment method if needed\n"
                        "3. Check your API usage and limits\n"
                        "4. Use a different API key if you have multiple accounts\n\n"
                        "Update your `.env` file with a valid API key and restart the app."
                    )
                elif "401" in error_msg or "invalid" in error_msg.lower():
                    st.error(
                        "❌ **Invalid API Key**\n\n"
                        "Your OpenAI API key is invalid or expired.\n\n"
                        "**Solutions:**\n"
                        "1. Get a new API key: https://platform.openai.com/api-keys\n"
                        "2. Update your `.env` file with the correct key\n"
                        "3. Restart the app"
                    )
                else:
                    st.error(f"❌ **Error generating answer:**\n\n{error_msg}")

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
