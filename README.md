# RAG PDF Chatbot

A complete **Retrieval-Augmented Generation (RAG)** system that lets you upload PDF files and ask questions about their content. The system uses embeddings, vector databases, and LLMs to provide accurate, context-grounded answers.

## 🎯 What is RAG?

**RAG (Retrieval-Augmented Generation)** combines three key components:

1. **Retrieval**: Find relevant information from documents
2. **Augmentation**: Add this information to the LLM prompt
3. **Generation**: LLM generates answers based on the context

This prevents hallucination and grounds answers in actual document content.

## 📊 The RAG Pipeline

```
PDF → Text → Chunks → Embeddings → Vector DB
                                        ↓
                            User Question → Embedding
                                        ↓
                            Similar Chunks ← Vector DB Search
                                        ↓
                            LLM (with context) → Answer
```

### Step-by-Step Explanation

| Step | Input | Process | Output | Why? |
|------|-------|---------|--------|------|
| 1 | PDF file | Extract text | Raw text | PDFs are binary, need readable text |
| 2 | Raw text | Split into chunks | Text chunks | LLMs have token limits |
| 3 | Chunks | Convert to vectors | Embeddings | Enable similarity search |
| 4 | Embeddings | Index in database | Vector DB | Fast retrieval of similar chunks |
| 5 | Question | Convert to vector | Question embedding | Match with chunk embeddings |
| 6 | Question embedding | Search vector DB | Top-k chunks | Find most relevant context |
| 7 | Chunks + Question | Send to LLM | Answer | Generate grounded response |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd rag-pdf-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Key

Create `.env` file in project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

Get your key from: https://platform.openai.com/api-keys

### 3. Run the Application

```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

## 📁 Project Structure

```
rag-pdf-chatbot/
├── src/
│   ├── pdf_loader.py          # PDF → Text extraction
│   ├── text_chunker.py        # Text → Chunks splitting
│   ├── embeddings_db.py       # Chunks → Embeddings → Vector DB
│   ├── rag_retriever.py       # Question → Retrieval → Answer
│   └── __init__.py
├── examples/
│   ├── basic_example.py       # Simple usage
│   └── advanced_example.py    # Advanced features
├── app.py                      # Streamlit UI
├── requirements.txt            # Dependencies
├── SETUP.md                    # Detailed setup guide
├── README.md                   # This file
└── .env                        # API keys (create this)
```

## 📚 Module Documentation

### 1. PDF Loader (`src/pdf_loader.py`)

Extracts text from PDF files.

```python
from src.pdf_loader import PDFLoader

loader = PDFLoader("document.pdf")
text = loader.load()  # Get all text
data = loader.load_with_metadata()  # Get text + metadata
```

**Key Points:**
- Uses PyPDF library for reliable extraction
- Preserves page structure
- Handles multi-page documents

---

### 2. Text Chunker (`src/text_chunker.py`)

Splits text into overlapping chunks.

```python
from src.text_chunker import RecursiveCharacterSplitter

chunker = RecursiveCharacterSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
chunks = chunker.split_text(text)
```

**Chunking Strategies:**

- **Character-based**: Simple, fast, may split sentences
- **Sentence-based**: Respects sentence boundaries
- **Recursive**: Splits by paragraphs → sentences → words → characters

**Why Overlap?**
```
Without overlap:
[Chunk 1: "...end of topic A"][Chunk 2: "start of topic B..."]
                              ^ Context lost!

With overlap:
[Chunk 1: "...end of topic A"][Chunk 2: "...topic A start of topic B..."]
                               ^ Context preserved!
```

---

### 3. Embeddings & Vector DB (`src/embeddings_db.py`)

Converts chunks to embeddings and stores in vector database.

```python
from src.embeddings_db import RAGVectorStore

vector_store = RAGVectorStore(model_name="all-MiniLM-L6-v2")
vector_store.add_documents(chunks)

# Search for similar chunks
results = vector_store.search("What is machine learning?", k=5)
```

**How Embeddings Work:**

```
Text: "The cat sat on the mat"
Embedding: [0.234, -0.567, 0.891, ..., 0.123]  (384 dimensions)

Text: "A cat was sitting on a mat"
Embedding: [0.245, -0.578, 0.902, ..., 0.134]  (similar!)

Distance between vectors = 0.05 (very similar)
```

**Vector Database (FAISS):**
- Stores all embeddings
- Enables fast nearest-neighbor search
- Can search 1000+ chunks in milliseconds

---

### 4. RAG Retriever (`src/rag_retriever.py`)

Retrieves relevant chunks and generates answers.

```python
from src.rag_retriever import RAGRetriever

retriever = RAGRetriever(vector_store, api_key)

# Simple retrieval
result = retriever.generate_answer("What is the main topic?", k=5)
print(result['answer'])
print(result['sources'])  # Chunks used for answer
```

**Advanced Features:**

```python
from src.rag_retriever import AdvancedRAGRetriever

advanced = AdvancedRAGRetriever(vector_store, api_key)

# Multi-query: Ask variations of the question
result = advanced.generate_answer_with_multi_query(
    "What is machine learning?",
    k=5
)

# Reranking: Retrieve more, use best ones
result = advanced.generate_answer_with_reranking(
    "What is machine learning?",
    k=10,      # Retrieve 10
    rerank_k=5 # Use best 5
)
```

## 🎛️ Configuration Guide

### Chunk Size
- **200-500 chars**: Specific, less context
- **800-1200 chars**: Balanced (recommended)
- **1500+ chars**: More context, less specific

### Chunk Overlap
- Prevents losing context at boundaries
- Typically 15-20% of chunk size
- Example: 1000 char chunks → 150-200 overlap

### Number of Retrieved Chunks (k)
- **3-5**: Fast, focused answers
- **5-10**: Comprehensive answers
- **10+**: Very thorough but slower

### Embedding Model
- **all-MiniLM-L6-v2**: Fast, good quality (default)
- **all-mpnet-base-v2**: Slower, better quality
- **all-distilroberta-v1**: Fast, decent quality

## 💡 Usage Examples

### Basic Usage

```python
from src.pdf_loader import PDFLoader
from src.text_chunker import RecursiveCharacterSplitter
from src.embeddings_db import RAGVectorStore
from src.rag_retriever import RAGRetriever
import os

# 1. Load PDF
loader = PDFLoader("document.pdf")
text = loader.load()

# 2. Chunk text
chunker = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200)
chunks = chunker.split_text(text)

# 3. Create embeddings
vector_store = RAGVectorStore()
vector_store.add_documents(chunks)

# 4. Answer questions
retriever = RAGRetriever(vector_store, os.getenv("OPENAI_API_KEY"))
result = retriever.generate_answer("What is the main topic?")
print(result['answer'])
```

### Save and Load Vector Store

```python
# Save
vector_store.save("my_vector_store")

# Load later
vector_store = RAGVectorStore()
vector_store.load("my_vector_store")
```

### Analyze Retrieval Results

```python
# See which chunks are retrieved
results = retriever.retrieve("What is AI?", k=5)

for i, (chunk, distance) in enumerate(results, 1):
    similarity = 1 / (1 + distance)
    print(f"Rank {i}: {similarity:.1%} similar")
    print(f"  {chunk[:100]}...")
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "OPENAI_API_KEY not found" | Create `.env` with your API key |
| Slow performance | Reduce chunk size or k value |
| Poor answer quality | Increase chunk size or k value |
| Out of memory | Reduce chunk size or process in batches |
| PDF text extraction fails | Check if PDF is text-based (not scanned image) |

## 📖 Learning Resources

- **LangChain**: https://python.langchain.com/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Sentence Transformers**: https://www.sbert.net/
- **OpenAI API**: https://platform.openai.com/docs/api-reference

## 🎓 Key Concepts

### Embeddings
Numerical representations of text meaning. Similar texts have similar embeddings.

### Vector Database
Stores embeddings and enables fast similarity search using algorithms like FAISS.

### Similarity Search
Finding the most similar vectors to a query vector using distance metrics (L2, cosine, etc.).

### Context Window
The amount of text (tokens) you can send to an LLM. RAG helps by sending only relevant chunks.

### Hallucination
When LLMs generate false information. RAG reduces this by grounding answers in actual documents.

## 📝 Next Steps

1. **Run the app**: `streamlit run app.py`
2. **Upload a PDF** and ask questions
3. **Explore examples**: Check `examples/` folder
4. **Customize**: Modify parameters for your use case
5. **Read SETUP.md**: Detailed step-by-step guide

## 📄 License

This project is open source and available under the MIT License.
