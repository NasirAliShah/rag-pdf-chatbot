# RAG PDF Chatbot - Setup Guide

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that combines:
1. **Retrieval**: Finding relevant information from documents
2. **Augmentation**: Adding this information to the LLM prompt
3. **Generation**: Using an LLM to generate answers based on the context

This prevents the LLM from hallucinating and grounds answers in actual document content.

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd rag-pdf-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

Get your API key from: https://platform.openai.com/api-keys

## Running the Application

### Using Streamlit (Recommended - Easy UI)
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Using Python Script (Programmatic)
```bash
python examples/basic_example.py
```

## Project Structure

```
rag-pdf-chatbot/
├── src/
│   ├── pdf_loader.py          # Step 1: PDF → Text
│   ├── text_chunker.py        # Step 2: Text → Chunks
│   ├── embeddings_db.py       # Step 3-4: Chunks → Embeddings → Vector DB
│   ├── rag_retriever.py       # Step 5-7: Question → Retrieval → Answer
│   └── __init__.py
├── examples/
│   ├── basic_example.py       # Simple usage example
│   └── advanced_example.py    # Advanced features
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
└── README.md                   # Project documentation
```

## Understanding the RAG Pipeline

### Step 1: PDF → Text
**File:** `src/pdf_loader.py`

Extracts text from PDF files using PyPDF library.

```python
from src.pdf_loader import PDFLoader

loader = PDFLoader("document.pdf")
text = loader.load()
```

**Why:** PDFs are binary files. We need to convert them to readable text.

---

### Step 2: Text → Chunks
**File:** `src/text_chunker.py`

Splits large text into smaller, overlapping chunks.

```python
from src.text_chunker import RecursiveCharacterSplitter

chunker = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200)
chunks = chunker.split_text(text)
```

**Why:** 
- LLMs have token limits (can't process entire documents)
- Smaller chunks are easier to embed and retrieve
- Overlap preserves context at chunk boundaries

**Example:**
```
Text: "The cat sat on the mat. The mat was red. The cat was happy."

Chunks (size=20, overlap=10):
- Chunk 1: "The cat sat on the "
- Chunk 2: "on the mat. The mat "
- Chunk 3: "mat was red. The cat"
- Chunk 4: "cat was happy."
```

---

### Step 3: Chunks → Embeddings
**File:** `src/embeddings_db.py`

Converts text chunks into numerical vectors (embeddings).

```python
from src.embeddings_db import RAGVectorStore

vector_store = RAGVectorStore()
vector_store.add_documents(chunks)
```

**Why:**
- Embeddings represent text meaning as numbers
- Similar texts have similar embeddings
- Enables fast similarity search

**How it works:**
```
Text: "The cat sat on the mat"
Embedding: [0.234, -0.567, 0.891, ..., 0.123]  (384 dimensions)

Text: "A cat was sitting on a mat"
Embedding: [0.245, -0.578, 0.902, ..., 0.134]  (similar!)
```

---

### Step 4: Embeddings → Vector Database
**File:** `src/embeddings_db.py`

Stores embeddings in FAISS for fast retrieval.

```python
# Already done in step 3 with RAGVectorStore
# FAISS automatically indexes embeddings for fast search
```

**Why:**
- Searching through 1000+ embeddings would be slow
- FAISS uses optimized algorithms for nearest neighbor search
- Can find similar chunks in milliseconds

---

### Step 5: User Question → Embedding
**File:** `src/rag_retriever.py`

Converts user's question to an embedding using the same model.

```python
question = "What is the main topic?"
question_embedding = embeddings_gen.encode_single(question)
```

**Why:** Must use the same embedding model as chunks for consistency.

---

### Step 6: Similar Chunks → Retrieval
**File:** `src/rag_retriever.py`

Finds the most relevant chunks using vector similarity.

```python
from src.rag_retriever import RAGRetriever

retriever = RAGRetriever(vector_store, api_key)
relevant_chunks = retriever.retrieve(question, k=5)
```

**Why:** Returns top-k most similar chunks to use as context.

---

### Step 7: Chunks → LLM → Answer
**File:** `src/rag_retriever.py`

Sends relevant chunks + question to LLM for answer generation.

```python
result = retriever.generate_answer(question, k=5)
print(result['answer'])
```

**Why:** LLM generates answers grounded in actual document content.

## Configuration Tips

### Chunk Size
- **Small (200-500 chars):** More specific, less context
- **Medium (800-1200 chars):** Balanced (recommended)
- **Large (1500+ chars):** More context, less specific

### Chunk Overlap
- Prevents losing context at boundaries
- Typically 15-20% of chunk size
- Example: 1000 char chunks → 150-200 overlap

### Number of Retrieved Chunks (k)
- **3-5:** Fast, focused answers
- **5-10:** Comprehensive answers
- **10+:** Very thorough but slower

## Troubleshooting

### "OPENAI_API_KEY not found"
- Create `.env` file with your API key
- Make sure `.env` is in the project root directory

### Slow Performance
- Reduce chunk size
- Reduce number of retrieved chunks (k)
- Use faster embedding model: "all-MiniLM-L6-v2" (default)

### Poor Answer Quality
- Increase chunk size (more context)
- Increase number of retrieved chunks (k)
- Try "Multi-Query" retrieval method
- Check if PDF text extraction worked properly

### Out of Memory
- Reduce chunk size
- Process PDFs in batches
- Use CPU-only FAISS (already default)

## Next Steps

1. **Run the Streamlit app:** `streamlit run app.py`
2. **Upload a PDF** and ask questions
3. **Explore examples:** Check `examples/` folder
4. **Customize:** Modify parameters in the sidebar
5. **Learn more:** Read the docstrings in each module

## Resources

- **LangChain Documentation:** https://python.langchain.com/
- **FAISS Documentation:** https://github.com/facebookresearch/faiss
- **Sentence Transformers:** https://www.sbert.net/
- **OpenAI API:** https://platform.openai.com/docs/api-reference
