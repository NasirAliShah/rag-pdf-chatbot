# Pinecone + Hugging Face Setup Guide

## Overview

This guide shows you how to use:
- **Pinecone** - Cloud vector database (persistent, scalable)
- **Hugging Face** - Embedding models (free, customizable)

Instead of:
- **FAISS** - Local vector database (temporary, limited)
- **SentenceTransformers** - Local embeddings

---

## Benefits

### Pinecone Advantages:
✅ **Persistent** - Data survives app restarts  
✅ **Cloud-based** - No local storage needed  
✅ **Scalable** - Handles millions of vectors  
✅ **Fast** - Optimized for similarity search  
✅ **Metadata filtering** - Advanced search capabilities  

### Hugging Face Advantages:
✅ **Free** - All models are open-source  
✅ **Variety** - Thousands of models to choose from  
✅ **Customizable** - Fine-tune for your use case  
✅ **Latest models** - Access to state-of-the-art embeddings  

---

## Step 1: Get Pinecone API Key

1. **Sign up** at https://www.pinecone.io/
2. **Create account** (free tier available)
3. **Get API key** from dashboard
4. **Copy API key**

**Free Tier:**
- 1 index
- 100K vectors
- 5GB storage
- Perfect for testing!

---

## Step 2: Add API Key to .env

Edit your `.env` file:

```bash
# OpenAI (optional)
OPENAI_API_KEY=sk-...your-openai-key...

# Pinecone (required for Pinecone mode)
PINECONE_API_KEY=your-pinecone-api-key-here
```

---

## Step 3: Install Dependencies

```bash
# Activate virtual environment
source venu/bin/activate  # Linux/Mac
# or
venu\Scripts\activate  # Windows

# Install new dependencies
pip install pinecone-client transformers tiktoken
```

---

## Step 4: Choose Embedding Model

Available Hugging Face models in the app:

| Model | Dimension | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| **all-MiniLM-L6-v2** | 384 | Fast | Good | General purpose (recommended) |
| **all-mpnet-base-v2** | 768 | Medium | Better | Higher quality needed |
| **bge-small-en-v1.5** | 384 | Fast | Good | Retrieval-optimized |
| **bge-base-en-v1.5** | 768 | Medium | Better | Best retrieval quality |

**Recommendation:** Start with `all-MiniLM-L6-v2` (default)

---

## Step 5: Run the App

```bash
streamlit run app.py
```

---

## Step 6: Configure in UI

### In Sidebar:

1. **Vector Database:** Select "Pinecone (Cloud, Paid)"
2. **Hugging Face Model:** Choose embedding model
3. **LLM Provider:** Choose Ollama or OpenAI
4. **Upload PDF** and ask questions!

---

## How It Works

### Data Flow with Pinecone:

```
1. PDF Upload
   └─> Extract text
   
2. Text Chunking
   └─> Split into chunks
   
3. Hugging Face Embeddings
   └─> Convert chunks to vectors using HF model
   
4. Pinecone Storage
   └─> Upload vectors to Pinecone cloud
   └─> Persistent storage (survives restarts)
   
5. Question Answering
   └─> Convert question to vector
   └─> Search Pinecone for similar chunks
   └─> Send to LLM for answer
```

---

## Pinecone Index Management

### View Index Stats

The app automatically shows:
- Number of vectors stored
- Index dimension
- Namespace info

### Clear Index

To delete all data:

```python
from src.pinecone_store import RAGPineconeStore

store = RAGPineconeStore()
store.clear()  # Deletes all vectors
```

### Multiple Documents

Pinecone stores all documents in the same index. To separate:

1. Use different namespaces (modify code)
2. Create multiple indexes (requires paid plan)

---

## Cost Comparison

### FAISS (Local):
- **Cost:** Free
- **Storage:** Local disk
- **Persistence:** Lost on restart
- **Scalability:** Limited by RAM

### Pinecone (Cloud):
- **Cost:** Free tier (100K vectors), then $70/month
- **Storage:** Cloud
- **Persistence:** Permanent
- **Scalability:** Millions of vectors

**For your resume PDF:**
- ~20 chunks = 20 vectors
- Well within free tier!

---

## Embedding Model Comparison

### SentenceTransformers (Current):
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
```

### Hugging Face Transformers (New):
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
```

**Difference:** 
- Same models, different libraries
- Hugging Face gives more control
- Can use ANY model from Hugging Face Hub

---

## Troubleshooting

### Error: "PINECONE_API_KEY not found"
**Solution:** Add API key to `.env` file

### Error: "Index already exists"
**Solution:** App reuses existing index (this is normal)

### Error: "Dimension mismatch"
**Solution:** Delete index and recreate:
```python
from pinecone import Pinecone
pc = Pinecone(api_key="your-key")
pc.delete_index("rag-pdf-chatbot")
```

### Slow embedding generation
**Solution:** First time downloads model from Hugging Face (1-2 min)

### Out of memory
**Solution:** Use smaller model (all-MiniLM-L6-v2) or reduce chunk size

---

## Advanced: Custom Embedding Models

Want to use a different Hugging Face model?

1. **Find model** on https://huggingface.co/models
2. **Check dimension** (usually in model card)
3. **Edit** `src/pinecone_store.py`:

```python
dimension_map = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "your-custom-model": 512,  # Add your model here
}
```

4. **Add to UI** in `app.py`:

```python
embedding_model = st.selectbox(
    "Hugging Face Embedding Model",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "your-custom-model"  # Add here
    ]
)
```

---

## OpenAI Model Selection

In the sidebar, when you select "OpenAI (Paid, Cloud)", you can now choose:

- **gpt-3.5-turbo** - Fastest, cheapest ($0.0005/1K tokens)
- **gpt-4** - Best quality ($0.03/1K tokens)
- **gpt-4-turbo** - Fast + quality ($0.01/1K tokens)

**Recommendation:** Use gpt-3.5-turbo for testing, gpt-4 for production

---

## Complete Setup Example

### .env file:
```bash
OPENAI_API_KEY=sk-proj-...
PINECONE_API_KEY=pcsk_...
```

### Run app:
```bash
streamlit run app.py
```

### Configure:
1. Vector Database: **Pinecone (Cloud, Paid)**
2. Embedding Model: **all-MiniLM-L6-v2**
3. LLM Provider: **OpenAI (Paid, Cloud)**
4. OpenAI Model: **gpt-3.5-turbo**
5. Upload PDF
6. Ask questions!

---

## Testing

### Test 1: Upload Resume
- Upload your resume PDF
- Check: "Embeddings stored in Pinecone" message
- Verify: Vectors are in Pinecone dashboard

### Test 2: Ask Questions
- "What is the person's CGPA?"
- "How many years of experience?"
- "What are the main skills?"

### Test 3: Restart App
- Close and restart Streamlit
- Data should still be in Pinecone
- (Note: You'll need to re-upload PDF to query it)

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Get Pinecone API key
3. ✅ Add to .env file
4. ✅ Run app and test
5. 📊 Monitor token usage (OpenAI)
6. 📊 Monitor vector count (Pinecone)
7. 🚀 Deploy to production!

---

## Resources

- **Pinecone Docs:** https://docs.pinecone.io/
- **Hugging Face Models:** https://huggingface.co/models
- **OpenAI Pricing:** https://openai.com/pricing
- **Pinecone Pricing:** https://www.pinecone.io/pricing/

---

## Summary

You now have:
- ✅ Pinecone cloud vector database
- ✅ Hugging Face embedding models
- ✅ OpenAI model selection (gpt-3.5, gpt-4)
- ✅ Token usage tracking
- ✅ Persistent storage
- ✅ Scalable architecture

**Your RAG system is production-ready!** 🚀
