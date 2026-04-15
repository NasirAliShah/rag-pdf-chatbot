# RAG Concepts - Deep Dive

This document provides detailed explanations of RAG concepts to help you understand how the system works.

## Table of Contents
1. [Embeddings](#embeddings)
2. [Vector Databases](#vector-databases)
3. [Similarity Search](#similarity-search)
4. [Chunking Strategies](#chunking-strategies)
5. [Retrieval Methods](#retrieval-methods)
6. [LLM Integration](#llm-integration)

---

## Embeddings

### What are Embeddings?

Embeddings are numerical representations of text meaning. They convert words, sentences, or documents into vectors (lists of numbers) that capture semantic meaning.

### How They Work

**Example:**
```
Text: "The cat sat on the mat"
Embedding: [0.234, -0.567, 0.891, 0.123, ..., 0.456]
           (384 dimensions for all-MiniLM-L6-v2 model)
```

### Key Properties

1. **Semantic Similarity**: Similar texts have similar embeddings
   ```
   "The cat sat on the mat" → [0.234, -0.567, 0.891, ...]
   "A cat was sitting on a mat" → [0.245, -0.578, 0.902, ...]
   
   Distance: 0.05 (very similar!)
   ```

2. **Dimensionality**: Each embedding has fixed dimensions
   - all-MiniLM-L6-v2: 384 dimensions
   - all-mpnet-base-v2: 768 dimensions
   - OpenAI text-embedding-3-small: 1536 dimensions

3. **Consistency**: Same text always produces same embedding

### Embedding Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose (recommended) |
| all-mpnet-base-v2 | 768 | Medium | Better | High-quality retrieval |
| all-distilroberta-v1 | 768 | Fast | Good | Fast with decent quality |
| text-embedding-3-small | 1536 | Medium | Excellent | OpenAI's latest |

### Why Embeddings Matter for RAG

Without embeddings, finding relevant chunks would require:
- Keyword matching (misses synonyms)
- Full text comparison (very slow)
- Manual tagging (not scalable)

With embeddings:
- Semantic search (understands meaning)
- Fast similarity calculation (milliseconds)
- Automatic relevance ranking

---

## Vector Databases

### What is a Vector Database?

A vector database stores embeddings and enables fast similarity search. Instead of comparing a query to every document, it uses optimized algorithms to find similar vectors quickly.

### FAISS (Facebook AI Similarity Search)

We use FAISS in this project. It's a library for efficient similarity search.

**How FAISS Works:**

```
1. Index Creation
   Embeddings: [e1, e2, e3, e4, e5, ...]
   ↓
   FAISS Index (optimized for fast search)

2. Search
   Query: q
   ↓
   Find k nearest neighbors
   ↓
   Return: [e2, e4, e1] (most similar)
```

### Distance Metrics

FAISS uses **L2 Distance** (Euclidean distance):

```
Distance = √[(x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²]

Lower distance = more similar
Higher distance = less similar
```

**Example:**
```
Query embedding: [0.5, 0.5]
Chunk 1 embedding: [0.6, 0.6] → Distance: 0.141 (similar)
Chunk 2 embedding: [0.1, 0.1] → Distance: 0.566 (different)
Chunk 3 embedding: [0.51, 0.51] → Distance: 0.014 (very similar)

Ranking: Chunk 3 > Chunk 1 > Chunk 2
```

### Saving and Loading

```python
# Save to disk
vector_store.save("my_store")
# Creates:
# - my_store/index.faiss (embeddings index)
# - my_store/chunks.pkl (text chunks)
# - my_store/embeddings.npy (numpy array of embeddings)

# Load from disk
vector_store.load("my_store")
```

---

## Similarity Search

### How Similarity Search Works

```
Step 1: Convert query to embedding
Query: "What is machine learning?"
↓
Query Embedding: [0.234, -0.567, 0.891, ...]

Step 2: Search vector database
↓
Find k nearest neighbors

Step 3: Return results
Results: [
  (chunk_1, distance=0.05),  # Most similar
  (chunk_2, distance=0.12),
  (chunk_3, distance=0.18),
  (chunk_4, distance=0.25),
  (chunk_5, distance=0.31)   # Least similar
]
```

### Similarity Score

Convert distance to similarity percentage:
```
Similarity = 1 / (1 + distance)

Distance 0.05 → Similarity 95%
Distance 0.12 → Similarity 89%
Distance 0.25 → Similarity 80%
Distance 1.00 → Similarity 50%
```

### Tuning k (Number of Results)

```
k=3: Fast, focused
  - Retrieves only most relevant chunks
  - Faster LLM processing
  - Risk: Missing important context

k=5: Balanced (recommended)
  - Good coverage of relevant information
  - Reasonable LLM processing time
  - Good balance of quality and speed

k=10: Comprehensive
  - Covers more context
  - Slower LLM processing
  - Risk: Too much irrelevant information
```

---

## Chunking Strategies

### Why Chunking?

**Problem:** LLMs have token limits
```
GPT-3.5-turbo: 4,096 tokens max
GPT-4: 8,192 or 32,768 tokens max

1 token ≈ 4 characters

So GPT-3.5 can handle ~16,000 characters max
A typical PDF has 100,000+ characters
```

**Solution:** Split into chunks

### Chunk Size Impact

**Small Chunks (200-500 chars)**
```
Pros:
- More specific information
- Faster processing
- Less noise

Cons:
- May lose context
- More chunks to manage
- May split important concepts
```

**Medium Chunks (800-1200 chars)**
```
Pros:
- Good balance of context and specificity
- Reasonable number of chunks
- Preserves most concepts

Cons:
- May include some irrelevant info
- Slightly slower processing
```

**Large Chunks (1500+ chars)**
```
Pros:
- Lots of context
- Fewer chunks overall
- Preserves full concepts

Cons:
- May include irrelevant information
- Slower processing
- Less specific retrieval
```

### Chunk Overlap

**Why Overlap?**

```
Without overlap:
[Chunk 1: "...end of topic A"]
[Chunk 2: "start of topic B..."]
                    ^ Context lost!

With 20% overlap (200 chars in 1000-char chunks):
[Chunk 1: "...end of topic A"]
[Chunk 2: "...topic A start of topic B..."]
                    ^ Context preserved!
```

**Overlap Percentage:**
- 10-15%: Minimal overlap, fewer chunks
- 15-20%: Balanced (recommended)
- 20-30%: Heavy overlap, more chunks

### Chunking Strategies

**1. Character-based**
```python
chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk_text(text)
```
- Simple and fast
- May split sentences
- Good for uniform content

**2. Sentence-based**
```python
chunks = chunker.chunk_text_by_sentences(text)
```
- Respects sentence boundaries
- Better for readability
- May create uneven chunk sizes

**3. Recursive (Recommended)**
```python
chunker = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200)
chunks = chunker.split_text(text)
```
- Tries multiple separators: paragraphs → sentences → words → characters
- Keeps related content together
- Best overall results

---

## Retrieval Methods

### Basic Retrieval

```python
results = retriever.retrieve(query, k=5)
# Returns: [(chunk_1, distance_1), (chunk_2, distance_2), ...]
```

**Process:**
1. Convert query to embedding
2. Search vector DB for k nearest neighbors
3. Return chunks sorted by similarity

### Multi-Query Retrieval

```python
result = retriever.generate_answer_with_multi_query(query, k=5)
```

**Process:**
1. Generate variations of the query
   - "What is machine learning?"
   - "Define machine learning"
   - "Explain machine learning"
   - "How does machine learning work?"

2. Retrieve chunks for each variation

3. Combine and deduplicate results

4. Use top-k for answer generation

**Advantages:**
- Catches different phrasings
- More comprehensive coverage
- Better for complex questions

**Disadvantages:**
- Slower (multiple searches)
- May retrieve redundant chunks

### Reranking Retrieval

```python
result = retriever.generate_answer_with_reranking(
    query,
    k=10,      # Initial retrieval
    rerank_k=5 # Final selection
)
```

**Process:**
1. Retrieve more chunks initially (k=10)
2. Rerank by relevance
3. Use top-k after reranking (rerank_k=5)

**Advantages:**
- Better coverage initially
- Reranking improves quality
- Balances recall and precision

**Disadvantages:**
- Slower than basic retrieval
- Reranking adds complexity

---

## LLM Integration

### Prompt Engineering for RAG

**Basic Prompt Structure:**
```
System: You are a helpful assistant that answers questions based on provided context.
        Always answer based on the context given. If the context doesn't contain 
        the answer, say so clearly.

User: Context from document:
      [Source 1] (Relevance: 95%)
      {chunk_1}
      
      [Source 2] (Relevance: 89%)
      {chunk_2}
      
      Question: {user_question}
      
      Please answer the question based on the context provided above.
```

### Why This Works

1. **System Message**: Sets expectations
   - Be helpful
   - Use context
   - Be honest about limitations

2. **Context Chunks**: Provides facts
   - Grounds answer in document
   - Prevents hallucination
   - Shows relevance scores

3. **Clear Question**: Focuses LLM
   - Explicit instruction
   - Clear task
   - Prevents confusion

### Temperature Setting

```python
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7  # Controls randomness
)
```

**Temperature Effects:**
- 0.0: Deterministic (same answer every time)
- 0.5: Balanced (some variation)
- 0.7: Default (good for RAG)
- 1.0: Creative (more variation)

**For RAG:** Use 0.5-0.7 (factual but natural)

### Handling No Results

```python
if not retrieved_chunks:
    return {
        'answer': "I couldn't find relevant information in the document.",
        'sources': [],
        'num_sources': 0
    }
```

Important to handle gracefully when no relevant chunks found.

---

## Complete RAG Flow Example

```
User: "What is the main topic?"

Step 1: Embed question
  Query: "What is the main topic?"
  ↓
  Query Embedding: [0.234, -0.567, 0.891, ...]

Step 2: Search vector DB
  ↓
  Find 5 most similar chunks
  ↓
  Results: [
    (chunk_1, distance=0.05),
    (chunk_2, distance=0.12),
    (chunk_3, distance=0.18),
    (chunk_4, distance=0.25),
    (chunk_5, distance=0.31)
  ]

Step 3: Format context
  Context = """
  [Source 1] (Relevance: 95%)
  {chunk_1_text}
  
  [Source 2] (Relevance: 89%)
  {chunk_2_text}
  
  ...
  """

Step 4: Create prompt
  Prompt = """
  System: You are a helpful assistant...
  
  User: Context from document:
  {context}
  
  Question: What is the main topic?
  
  Please answer based on the context above.
  """

Step 5: Send to LLM
  ↓
  LLM generates answer based on context

Step 6: Return result
  {
    'answer': 'The main topic is...',
    'sources': [chunk_1, chunk_2, ...],
    'num_sources': 5
  }
```

---

## Performance Optimization

### Faster Retrieval

1. **Reduce chunk size**: Fewer chunks to search
2. **Reduce k**: Retrieve fewer results
3. **Use faster embedding model**: all-MiniLM-L6-v2
4. **Batch processing**: Process multiple queries together

### Better Quality

1. **Increase chunk size**: More context
2. **Increase k**: More comprehensive
3. **Use better embedding model**: all-mpnet-base-v2
4. **Multi-query retrieval**: Better coverage
5. **Reranking**: Better relevance

### Memory Optimization

1. **Reduce chunk size**: Fewer embeddings
2. **Use CPU-only FAISS**: Already default
3. **Process in batches**: Don't load everything at once
4. **Delete unused vector stores**: Free up space

---

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Poor answer quality | Chunks too small | Increase chunk size |
| Missing context | Not enough chunks retrieved | Increase k |
| Slow performance | Too many chunks | Reduce chunk size or k |
| Hallucination | LLM not grounded | Ensure relevant chunks retrieved |
| Out of memory | Too many embeddings | Reduce chunk size |
| Wrong chunks retrieved | Bad embeddings | Use better embedding model |

