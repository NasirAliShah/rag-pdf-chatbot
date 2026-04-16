# Quick Start Guide - 5 Minutes to RAG

Get your RAG system running in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
cd rag-pdf-chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Set Up API Key (1 minute)

Create `.env` file in project root:
```
OPENAI_API_KEY=sk-your-key-here
```

Get your key: https://platform.openai.com/api-keys

## Step 3: Run the App (2 minutes)

```bash
streamlit run app.py
```

Open: http://localhost:8501

## Step 4: Upload and Ask!

1. Click "Choose a PDF file"
2. Select your PDF
3. Type your question
4. Click "Search & Answer"
5. Get your answer!

---

## What Just Happened?

Your RAG system executed this pipeline:

```
PDF File
   ↓
[PDF Loader] → Extract text
   ↓
[Text Chunker] → Split into 1000-char chunks with 200-char overlap
   ↓
[Embeddings] → Convert chunks to 384-dimensional vectors
   ↓
[Vector DB] → Store in FAISS for fast search
   ↓
Your Question
   ↓
[Embeddings] → Convert question to same 384-dimensional vector
   ↓
[Vector Search] → Find 5 most similar chunks
   ↓
[LLM Prompt] → Send chunks + question to GPT-3.5-turbo
   ↓
Answer
```

---

## Configuration Tips

### In the Sidebar:

- **Chunk Size**: 1000 (balanced) - increase for more context, decrease for specificity
- **Chunk Overlap**: 200 (20% of chunk size) - prevents losing context at boundaries
- **Number of Results**: 5 (balanced) - use 3 for speed, 10 for comprehensiveness

### Retrieval Method:

- **Standard**: Fast, good for simple questions
- **Multi-Query**: Slower but better for complex questions (generates question variations)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "OPENAI_API_KEY not found" | Create `.env` with your API key |
| Slow performance | Reduce chunk size or number of results |
| Poor answers | Increase chunk size or number of results |
| PDF text extraction fails | Ensure PDF is text-based (not scanned image) |

---

## Next Steps

1. **Explore the code**: Check `src/` folder to understand each component
2. **Read SETUP.md**: Detailed step-by-step guide
3. **Read CONCEPTS.md**: Deep dive into RAG concepts
4. **Try examples**: Run `python examples/basic_example.py`
5. **Customize**: Modify parameters for your use case

---

## File Structure

```
rag-pdf-chatbot/
├── app.py                  ← Main Streamlit app (run this!)
├── src/
│   ├── pdf_loader.py       ← Load PDFs
│   ├── text_chunker.py     ← Split text
│   ├── embeddings_db.py    ← Create embeddings
│   └── rag_retriever.py    ← Retrieve & answer
├── examples/
│   ├── basic_example.py    ← Simple usage
│   └── advanced_example.py ← Advanced features
├── requirements.txt        ← Dependencies
├── .env                    ← API keys (create this)
├── README.md               ← Full documentation
├── SETUP.md                ← Detailed setup
├── CONCEPTS.md             ← RAG concepts
└── QUICKSTART.md           ← This file
```

---

## Key Concepts (30-second version)

**RAG = Retrieval-Augmented Generation**

Instead of asking LLM directly:
```
Question → LLM → Answer (may hallucinate)
```

RAG does:
```
Question → Find relevant chunks → LLM (with context) → Answer (grounded in facts)
```

**Why it matters:**
- Prevents hallucination (LLM makes up stuff)
- Grounds answers in actual documents
- Provides source citations
- Works with any document

---

## Common Questions

**Q: How does it know which chunks are relevant?**
A: It converts both the question and chunks to numerical vectors (embeddings) and finds the most similar ones using math.

**Q: Why split into chunks?**
A: LLMs have token limits. A 100-page PDF won't fit in one prompt. Chunks let us send only relevant parts.

**Q: Can I use my own LLM?**
A: Yes! Modify `src/rag_retriever.py` to use a different LLM (Claude, Llama, etc.)

**Q: How accurate is it?**
A: As accurate as your PDF content. If the PDF has the answer, RAG will find it. If not, it will say so.

**Q: Can I save my vector store?**
A: Yes! Use `vector_store.save("path")` and `vector_store.load("path")`

---

## Performance Benchmarks

On a typical laptop with a 50-page PDF:

| Operation | Time |
|-----------|------|
| Load PDF | 0.5s |
| Create chunks | 0.1s |
| Generate embeddings | 5-10s |
| Answer question | 2-3s |
| **Total** | **~10-15s** |

---

## Next: Run Your First Query

```bash
streamlit run app.py
```

Then:
1. Upload a PDF
2. Ask "What is the main topic?"
3. Watch the RAG pipeline work!

Enjoy! 🚀
