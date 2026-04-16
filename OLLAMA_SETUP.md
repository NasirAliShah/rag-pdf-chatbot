# Using Ollama with RAG PDF Chatbot

This guide explains how to use Ollama (free, local LLM) instead of OpenAI with your RAG system.

## What is Ollama?

**Ollama** is a free, open-source tool that lets you run large language models locally on your computer. No API keys, no costs, no internet required.

**Advantages:**
- ✅ Completely free
- ✅ Runs locally (privacy)
- ✅ No API costs
- ✅ Works offline
- ✅ Perfect for testing and learning

**Disadvantages:**
- ⚠️ Slower than cloud APIs (depends on your hardware)
- ⚠️ Requires more disk space (~5-10GB per model)
- ⚠️ Quality depends on model size

## Installation

### Step 1: Install Ollama

**On macOS:**
```bash
# Download from https://ollama.ai
# Or use Homebrew
brew install ollama
```

**On Linux:**
```bash
curl https://ollama.ai/install.sh | sh
```

**On Windows:**
Download from https://ollama.ai/download

### Step 2: Start Ollama Server

Open a terminal and run:
```bash
ollama serve
```

You should see output like:
```
2024-04-16 10:30:45.123 [main] listening on 127.0.0.1:11434
```

**Keep this terminal open** - Ollama needs to be running while you use the RAG app.

### Step 3: Download a Model

In a **new terminal**, download a model:

**Recommended (Fast & Good Quality):**
```bash
ollama pull mistral
```

**Other Options:**
```bash
ollama pull llama2          # Larger, better quality
ollama pull neural-chat     # Optimized for chat
ollama pull dolphin-mixtral # Very capable
```

This downloads the model (5-10GB depending on the model). It's a one-time download.

### Step 4: Verify Installation

Test if Ollama is working:
```bash
curl http://localhost:11434/api/tags
```

You should see a list of downloaded models.

## Using with RAG PDF Chatbot

### Step 1: Start Ollama Server

In one terminal:
```bash
ollama serve
```

### Step 2: Start the RAG App

In another terminal:
```bash
cd /path/to/rag-pdf-chatbot
source venu/bin/activate
streamlit run app.py
```

### Step 3: Configure in the App

1. Open http://localhost:8501
2. In the sidebar, select **"Ollama (Free, Local)"**
3. Choose your model (e.g., "mistral")
4. Leave the URL as `http://localhost:11434`
5. Upload a PDF and ask questions!

## Model Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| mistral | 7B | Fast | Good | ⭐ Recommended |
| llama2 | 7B | Medium | Good | General use |
| neural-chat | 7B | Fast | Good | Chat |
| dolphin-mixtral | 46B | Slow | Excellent | Complex questions |

**Recommendation:** Start with `mistral` - it's fast and good quality.

## Troubleshooting

### Error: "Cannot connect to Ollama at http://localhost:11434"

**Solution:** Make sure Ollama server is running:
```bash
ollama serve
```

### Error: "Model not found"

**Solution:** Download the model first:
```bash
ollama pull mistral
```

### Slow Responses

**Causes:**
- Your computer is slow (Ollama needs good CPU/GPU)
- Model is too large (try a smaller model like mistral)
- Disk is slow (use SSD if possible)

**Solutions:**
- Use a smaller model (mistral instead of dolphin-mixtral)
- Close other applications
- Increase chunk size in the app to reduce processing

### High Memory Usage

Ollama loads the entire model into RAM. If you run out of memory:
- Use a smaller model
- Close other applications
- Increase system swap space

## Advanced: Using GPU Acceleration

Ollama can use GPU for faster inference. Check if your GPU is supported:

**NVIDIA GPU:**
```bash
# Install CUDA toolkit
# Ollama will automatically detect and use it
ollama serve
```

**AMD GPU:**
```bash
# Install ROCm
# Ollama will automatically detect and use it
ollama serve
```

**Apple Silicon (M1/M2/M3):**
Ollama automatically uses Metal acceleration.

## Comparing Ollama vs OpenAI

| Feature | Ollama | OpenAI |
|---------|--------|--------|
| Cost | Free | $0.0005 per 1K tokens |
| Speed | Slow (depends on hardware) | Fast |
| Quality | Good (7B models) | Excellent (GPT-3.5/4) |
| Privacy | Local (private) | Cloud (shared) |
| Internet | Not required | Required |
| Setup | Complex | Simple (API key) |

## Next Steps

1. **Install Ollama** following the steps above
2. **Start the server** with `ollama serve`
3. **Download a model** with `ollama pull mistral`
4. **Run the RAG app** and select Ollama in the sidebar
5. **Upload a PDF** and start asking questions!

## Resources

- **Ollama Website:** https://ollama.ai
- **Model Library:** https://ollama.ai/library
- **GitHub:** https://github.com/ollama/ollama
- **Discord Community:** https://discord.gg/ollama

## Tips for Best Results

1. **Use mistral model** - Good balance of speed and quality
2. **Adjust chunk size** - Larger chunks = more context but slower
3. **Use fewer results** - Start with 3-5 chunks, increase if needed
4. **Keep PDFs focused** - Smaller, focused documents work better
5. **Be specific with questions** - More specific questions get better answers

Happy chatting! 🚀
