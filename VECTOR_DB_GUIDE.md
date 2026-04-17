# Vector Database Visual Guide

## How Vector Database Works in RAG System

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                                │
└─────────────────────────────────────────────────────────────────┘

1. PDF UPLOAD
   └─> app.py (line 114-118)
       ↓
       PDFLoader.load_with_metadata()
       ↓
       Returns: {'text': '...', 'num_pages': 2, ...}

2. TEXT CHUNKING
   └─> app.py (line 150-154)
       ↓
       RecursiveCharacterSplitter.split_text()
       ↓
       Returns: ['chunk1', 'chunk2', 'chunk3', ...]

3. VECTOR DATABASE CREATION & POPULATION
   └─> app.py (line 180-181)
       ↓
       vector_store = RAGVectorStore(model_name="all-MiniLM-L6-v2")
       vector_store.add_documents(chunks)
       
       ┌──────────────────────────────────────────┐
       │   embeddings_db.py                       │
       │                                          │
       │   RAGVectorStore                         │
       │   ├─ EmbeddingsGenerator                 │
       │   │  └─ SentenceTransformer model        │
       │   │     (converts text → vectors)        │
       │   │                                      │
       │   └─ VectorDatabase (FAISS)              │
       │      └─ index.faiss (stores vectors)     │
       │      └─ chunks.pkl (stores text)         │
       │      └─ embeddings.npy (stores arrays)   │
       └──────────────────────────────────────────┘

4. QUESTION ANSWERING (RETRIEVAL)
   └─> app.py (line 204-242)
       ↓
       User asks: "What is the person's CGPA?"
       ↓
       retriever.generate_answer(question)
       
       ┌──────────────────────────────────────────┐
       │   rag_retriever.py                       │
       │                                          │
       │   RAGRetriever.retrieve()                │
       │   └─ vector_store.search(question)       │
       │      ├─ Encode question to vector        │
       │      ├─ Find similar chunks in FAISS     │
       │      └─ Return top k chunks              │
       │                                          │
       │   RAGRetriever.generate_answer()         │
       │   └─ Format context from chunks          │
       │   └─ Send to LLM with context            │
       │   └─ LLM extracts answer from context    │
       └──────────────────────────────────────────┘

5. ANSWER DISPLAY
   └─> app.py (line 243-265)
       ↓
       Display answer + sources
```

---

## Key Components

### 1. **EmbeddingsGenerator** (`embeddings_db.py:36-79`)
**What it does:** Converts text to numerical vectors

```python
# Location: embeddings_db.py
class EmbeddingsGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        # Converts list of chunks to embeddings
        # Returns: (num_chunks, 384) dimensional array
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        # Converts single query to embedding
        # Returns: (384,) dimensional array
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
```

**Example:**
```
Input:  "CGPA: 3.60/4.00"
Output: [0.234, -0.567, 0.891, ..., 0.456]  (384 numbers)
```

---

### 2. **VectorDatabase (FAISS)** (`embeddings_db.py:82-173`)
**What it does:** Stores embeddings and performs similarity search

```python
# Location: embeddings_db.py
class VectorDatabase:
    def __init__(self, embedding_dim=384):
        self.index = faiss.IndexFlatL2(embedding_dim)  # FAISS index
        self.chunks = []  # Stores original text chunks
        self.embeddings = None  # Stores embedding arrays
    
    def add_chunks(self, chunks: List[str], embeddings: np.ndarray):
        # ADDING DATA TO VECTOR DB
        self.chunks.extend(chunks)  # Store text
        self.index.add(embeddings)  # Store vectors
    
    def search(self, query_embedding: np.ndarray, k=5):
        # RETRIEVING DATA FROM VECTOR DB
        distances, indices = self.index.search(query_embedding, k)
        # Returns: [(chunk_text, distance), ...]
```

**Visual Example:**
```
Vector Space (simplified 2D):

Chunk 1: "CGPA: 3.60/4.00"        ●  ← Query: "What is CGPA?"  ◆
Chunk 2: "Bachelor of CS"         ●
Chunk 3: "5 years experience"     ●

FAISS finds the closest chunk (●) to query (◆)
Distance = how similar (lower = more similar)
```

---

### 3. **RAGVectorStore** (`embeddings_db.py:176-223`)
**What it does:** Combines embeddings generator + vector database

```python
# Location: embeddings_db.py
class RAGVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embeddings_gen = EmbeddingsGenerator(model_name)
        self.vector_db = VectorDatabase(embedding_dim=384)
    
    def add_documents(self, chunks: List[str]):
        # STEP 1: Generate embeddings for all chunks
        embeddings = self.embeddings_gen.encode(chunks)
        # STEP 2: Add to vector database
        self.vector_db.add_chunks(chunks, embeddings)
    
    def search(self, query: str, k=5):
        # STEP 1: Generate embedding for query
        query_embedding = self.embeddings_gen.encode_single(query)
        # STEP 2: Search vector database
        return self.vector_db.search(query_embedding, k)
```

---

## Where Vector Database is Used in App

### **Step 1: Creating Vector Database** (`app.py:180-181`)
```python
# When PDF is uploaded and chunks are created
vector_store = RAGVectorStore(model_name="all-MiniLM-L6-v2")
vector_store.add_documents(chunks)  # ← ADDING DATA HERE
```

**What happens:**
1. Takes 8 chunks from your resume
2. Converts each chunk to 384-dimensional vector
3. Stores vectors in FAISS index
4. Stores original text in memory

**Files created (in memory, not on disk):**
- `vector_store.embeddings_gen.model` - SentenceTransformer model
- `vector_store.vector_db.index` - FAISS index with 8 vectors
- `vector_store.vector_db.chunks` - List of 8 text chunks

---

### **Step 2: Retrieving from Vector Database** (`app.py:227-242`)
```python
# When user asks a question
question = "What is the person's CGPA?"

result = retriever.generate_answer(question)
```

**What happens inside `rag_retriever.py:retrieve()`:**
```python
def retrieve(self, query: str, k: int = 5):
    # STEP 1: Convert question to embedding
    query_embedding = self.vector_store.embeddings_gen.encode_single(query)
    
    # STEP 2: Search vector database
    results = self.vector_store.vector_db.search(query_embedding, k)
    # Returns: [
    #   ('CGPA: 3.60/4.00 University of Engineering...', 0.15),
    #   ('Bachelor of Computer Science CGPA: 3.60/4.00', 0.18),
    #   ...
    # ]
    
    return results
```

---

## Data Flow Example: "What is CGPA?"

```
1. USER QUESTION
   Input: "What is the person's CGPA?"

2. EMBEDDING GENERATION
   Question → SentenceTransformer → [0.123, -0.456, 0.789, ...]

3. SIMILARITY SEARCH IN FAISS
   Compare question embedding with all 8 chunk embeddings
   
   Chunk 1: "CGPA: 3.60/4.00..."     Distance: 0.15 ✓ (closest)
   Chunk 2: "Bachelor of CS..."      Distance: 0.18
   Chunk 3: "5 years experience..."  Distance: 0.45
   Chunk 4: "Blockchain skills..."   Distance: 0.52
   Chunk 5: "Tools & Libraries..."   Distance: 0.61
   ...

4. TOP K RESULTS RETURNED
   Returns top 5 most similar chunks with their distances

5. CONTEXT FORMATTING
   Combine top chunks into context:
   "
   [Source 1] (Relevance: 87%)
   CGPA: 3.60/4.00
   University of Engineering and Technology, Peshawar
   September 2016 - October 2020
   
   [Source 2] (Relevance: 85%)
   Bachelor of Computer Science
   ...
   "

6. LLM ANSWER GENERATION
   System: "Extract information from context"
   User: "Question: What is CGPA?\n\nContext: [above]\n\nAnswer:"
   
   LLM Response: "The person's CGPA is 3.60/4.00"

7. DISPLAY
   Answer: "The person's CGPA is 3.60/4.00"
   Sources Used: 5
   [Show sources...]
```

---

## Vector Database Files (If Saved to Disk)

If you save the vector database:
```python
vector_store.save("vector_store_backup/")
```

Files created:
```
vector_store_backup/
├── index.faiss          # FAISS index (binary format)
├── chunks.pkl           # Pickled list of text chunks
└── embeddings.npy       # NumPy array of embeddings
```

Load later:
```python
vector_store = RAGVectorStore()
vector_store.load("vector_store_backup/")
```

---

## Key Points

1. **Vector Database is IN-MEMORY** - Created when PDF is uploaded, deleted when app restarts
2. **FAISS Index** - Stores 384-dimensional vectors for fast similarity search
3. **Chunks Storage** - Original text chunks stored alongside vectors
4. **Search Speed** - FAISS is very fast (O(log n) with proper indexing)
5. **Similarity Metric** - Uses L2 distance (Euclidean distance)
   - Lower distance = more similar
   - Higher distance = less similar

---

## How to Debug Vector Database

Add this to `app.py` after creating vector_store:

```python
# Debug: Print vector database info
st.write(f"Vector DB Size: {vector_store.vector_db.index.ntotal} chunks")
st.write(f"Embedding Dimension: {vector_store.vector_db.embedding_dim}")
st.write(f"Chunks stored: {len(vector_store.vector_db.chunks)}")

# Test search
test_query = "CGPA"
results = vector_store.search(test_query, k=3)
st.write("Search results for 'CGPA':")
for chunk, distance in results:
    st.write(f"Distance: {distance:.3f} - {chunk[:100]}...")
```

This will show you exactly what's being stored and retrieved!
